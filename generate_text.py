"""
Text generation script for TRM.
Supports greedy, sampling, and ACT-based adaptive generation.
"""

import torch
import json
from transformers import AutoTokenizer
from pathlib import Path
import argparse

# CPU optimization: limit threads to avoid overload
torch.set_num_threads(4)  # Use half of threads for Intel i5-1035G1
torch.set_num_interop_threads(2)  # Reduce inter-op parallelism

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained TRM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint directory
    config_path = Path(checkpoint_path).parent / "all_config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Create model
    model = TinyRecursiveReasoningModel_ACTV1(config['arch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Check if using concept vocabulary
    use_concepts = hasattr(model, 'capsule_encoder') and model.capsule_encoder is not None
    
    # Load concept decoder if needed
    decoder = None
    if use_concepts:
        from models.concept_decoder import ConceptDecoder
        num_concepts = getattr(model.config, 'num_concepts', 2048)
        expansion_table_path = Path(checkpoint_path).parent / "concept_expansions.json"
        decoder = ConceptDecoder(
            num_concepts=num_concepts,
            expansion_table_path=str(expansion_table_path) if expansion_table_path.exists() else None
        )
        print(f"\n📚 Concept vocabulary: {num_concepts} concepts")
        if expansion_table_path.exists():
            print(f"   Expansion table loaded: {len(decoder.expansion_table.expansions)} mappings")
    
    return model, config, decoder


def generate_text(
    model,
    tokenizer,
    decoder,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    use_act: bool = True,
    use_capsules: bool = False,
    device: str = "cuda"
):
    """
    Generate text using TRM with adaptive computation.
    
    Args:
        model: TRM model
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (1.0 = default, <1 = more focused)
        top_p: Nucleus sampling threshold
        use_act: Use ACT halting (adaptive cycles per token)
        use_capsules: Use HESC capsule encoding
        device: cuda or cpu
    """
    # Encode prompt (capsule or token mode)
    if use_capsules and hasattr(model, 'capsule_encoder'):
        # HESC mode: encode to capsules
        capsule_data = model.encode_text([prompt], return_children=True)
        input_embeddings = capsule_data['sketches'].to(device)  # [1, k, D]
    else:
        # Token mode
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        input_embeddings = input_ids
    
    # Initialize carry
    batch = {
        "inputs": input_ids,
        "labels": input_ids,  # Dummy labels
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device)
    }
    carry = model.initial_carry(batch)
    
    generated_tokens = input_ids.tolist()[0]
    act_steps = []  # Track ACT cycles per token
    
    # Use inference_mode for better memory efficiency (vs no_grad)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Update batch with current sequence
            current_ids = torch.tensor([generated_tokens], device=device)
            batch_current = {
                "inputs": current_ids,
                "labels": current_ids,
                "puzzle_identifiers": torch.zeros(1, dtype=torch.int32, device=device)
            }
            
            # Forward pass with ACT
            carry_out, outputs = model(carry, batch_current)
            
            # Get logits for last token
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            # Handle concept vocabulary vs BPE
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'total_vocab'):
                # Concept vocabulary mode
                concept_id = torch.argmax(logits, dim=-1).item()
                generated_tokens.append(concept_id)
                
                # Check for control symbols
                if concept_id >= model.lm_head.num_concepts:
                    control_type = model.lm_head.decode_token(concept_id)
                    if '<STOP>' in control_type:
                        break
            else:
                # BPE token mode
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    
                    # Top-p sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()
                    
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_tokens.append(next_token.item())
            
            # Track ACT steps
            if use_act:
                steps = carry_out.steps[0].item()
                act_steps.append(steps)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Update carry for next iteration
            carry = carry_out
    
    # Decode based on mode
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'total_vocab'):
        # Concept vocabulary mode - use ConceptDecoder
        if decoder is None:
            print("Warning: No decoder provided for concept mode, returning concept IDs")
            generated_text = f"Concept IDs: {generated_tokens}"
        else:
            # Build control symbols and expand mask tensors
            concept_ids_tensor = torch.tensor([generated_tokens], device=device)
            control_symbols = (concept_ids_tensor >= model.lm_head.num_concepts).long()
            expand_mask = torch.zeros_like(concept_ids_tensor, dtype=torch.bool)
            
            # Decode concept sequence to text
            decoded_texts = decoder.decode_sequence(
                concept_ids_tensor,
                control_symbols,
                expand_mask,
                tokenizer
            )
            generated_text = decoded_texts[0] if decoded_texts else ""
    else:
        # BPE token mode - standard decoding
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Stats
    if use_act and act_steps:
        avg_steps = sum(act_steps) / len(act_steps)
        print(f"\nACT Stats: Avg {avg_steps:.1f} steps/token, Total: {sum(act_steps)} steps")
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with TRM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--use_act", action="store_true", help="Use ACT halting")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load model and decoder
    print(f"Loading model from {args.checkpoint}...")
    model, config, decoder = load_model(args.checkpoint, args.device)
    
    # Load tokenizer (for BPE decoding even in concept mode)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Detect mode
    use_capsules = hasattr(model, 'capsule_encoder') and model.capsule_encoder is not None
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Mode: {'Concept Vocabulary' if use_capsules else 'BPE Tokens'}")
    print(f"\nGenerating...")
    
    generated = generate_text(
        model,
        tokenizer,
        decoder,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_act=args.use_act,
        use_capsules=use_capsules,
        device=args.device
    )
    
    print(f"\n{'='*70}")
    print(f"Generated Text:")
    print(f"{'='*70}")
    print(generated)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
