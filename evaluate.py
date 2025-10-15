from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
# 'evaluate' 함수를 직접 수정해야 하므로 pretrain에서 가져오지 않습니다.
from pretrain import PretrainConfig, init_train_state, create_dataloader
# <--- 추가된 부분: tqdm을 가져옵니다.
from tqdm import tqdm
# <--- 추가된 부분: evaluate 함수에 필요한 다른 요소들을 가져옵니다.
from utils.functions import all_gather_object
import collections

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


# <--- 수정된 부분: pretrain.py에 있던 evaluate 함수를 이곳으로 가져와 직접 수정합니다.
@torch.no_grad()
def evaluate(config: PretrainConfig, train_state, loader, metadata, rank, world_size):
    all_metrics = collections.defaultdict(list)
    all_outputs = []

    # Model in eval mode
    train_state.model.eval()

    for batch in tqdm(loader, desc=f"Evaluating (Set: {config.data_test_set.split('/')[-1]})", disable=rank!=0):
        # <--- 가장 중요한 수정 부분 ---
        # 배치(batch)의 모든 텐서를 GPU로 이동시킵니다.
        batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # ---------------------------

        carry = train_state.model.initial_carry(batch)

        while True:
            carry, loss, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=set(config.eval_save_outputs))
            if all_finish:
                break

        for k, v in metrics.items():
            all_metrics[k].append(v)
        
        if len(config.eval_save_outputs) > 0:
            all_outputs.append(preds)

    if world_size > 1:
        all_metrics = {k: all_gather_object(v) for k, v in all_metrics.items()}
        if len(config.eval_save_outputs) > 0:
            all_outputs = all_gather_object(all_outputs)
    
    if rank == 0:
        metrics = {k: torch.cat(v, dim=0).mean().item() for k, v in all_metrics.items()}
        if len(config.eval_save_outputs) > 0:
            # Concat results
            outputs = {k: torch.cat([item[k] for item in all_outputs], dim=0) for k in config.eval_save_outputs}
            torch.save(outputs, os.path.join(config.checkpoint_path, f"outputs_{train_state.step}.pt"))
        return metrics
    else:
        return None


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))

    RANK = 0
    WORLD_SIZE = 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True, strict=False)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, assign=True, strict=False)

    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    print("Starting evaluation")
    train_state.model.eval()
    
    # 이제 수정된 evaluate 함수를 호출합니다.
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    launch()