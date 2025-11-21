import os
import torch
import logging

logger = logging.getLogger(__name__)

def setup_training_env():
    """
    Configures the training environment variables and PyTorch settings.
    Should be called at the very beginning of the training script.
    """
    # Memory optimization: Reduce CUDA memory fragmentation and enable expandable segments
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128'
    
    # Explicitly disable CUDA graphs to save ~1.5GB memory
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'
    
    print("\n" + "="*70)
    print("DEBUG: Environment Variables Set")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
    print(f"  TORCHINDUCTOR_CUDAGRAPHS = {os.environ.get('TORCHINDUCTOR_CUDAGRAPHS', 'NOT SET')}")
    print(f"  TORCHINDUCTOR_COMPILE_THREADS = {os.environ.get('TORCHINDUCTOR_COMPILE_THREADS', 'NOT SET')}")
    print("="*70 + "\n")
    
    # Enable memory-efficient attention backend (40-60% memory reduction for attention)
    if torch.cuda.is_available():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            # Force memory-efficient backend (works on T4, doesn't require Ampere)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)  # T4 doesn't support Flash
            torch.backends.cuda.enable_math_sdp(False)   # Math is slower and uses more memory
            logger.info("Enabled memory-efficient SDPA backend")
        except (ImportError, AttributeError):
            logger.warning("Could not enable memory-efficient SDPA backend (older PyTorch?)")
