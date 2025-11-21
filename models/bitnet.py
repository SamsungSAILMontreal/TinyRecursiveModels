import torch
import torch.nn as nn
import torch.nn.functional as F

class BitwiseQuantization(torch.autograd.Function):
    """
    Autograd function for 1.58-bit quantization with Straight-Through Estimator (STE).
    """
    @staticmethod
    def forward(ctx, input):
        # Scale by mean absolute value
        scale = 1.0 / input.abs().mean().clamp_(min=1e-5)
        # Quantize to {-1, 0, 1}
        # We divide by scale to normalize, round, clamp, then multiply back?
        # No, the paper says: W_quant = Round(W * scale) / scale? 
        # Actually usually it's: W_quant = Round(W / beta) * beta where beta is mean(abs(W)).
        # Let's follow the standard:
        # W_scaled = W / mean(abs(W))
        # W_quant = Round(W_scaled).clamp(-1, 1)
        # W_dequant = W_quant * mean(abs(W))
        
        ctx.scale = scale
        result = (input * scale).round().clamp_(-1, 1) / scale
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through unchanged
        # Optionally we could mask gradients where weights are clipped, but standard STE is identity.
        return grad_output

class ActivationQuantization(torch.autograd.Function):
    """
    Autograd function for 8-bit activation quantization with STE.
    """
    @staticmethod
    def forward(ctx, input):
        # Per-token quantization to 8 bits
        # Scale = 127 / max(abs(x))
        scale = 127.0 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        result = (input * scale).round().clamp_(-128, 127) / scale
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear(nn.Linear):
    """
    BitNet b1.58 Linear Layer.
    Replaces nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # Initialize weights? Standard initialization is fine, they get quantized on the fly.

    def forward(self, input):
        # 1. Quantize Input (8-bit)
        input_quant = ActivationQuantization.apply(input)
        
        # 2. Quantize Weights (1.58-bit)
        weight_quant = BitwiseQuantization.apply(self.weight)
        
        # 3. Linear operation
        # Note: In a real BitNet kernel, this would be efficient. 
        # Here we use fp32/bf16 matmul with quantized values (Simulated Quantization).
        return F.linear(input_quant, weight_quant, self.bias)

def replace_linear_with_bitlinear(model, exclude_names=None):
    """
    Recursively replace nn.Linear with BitLinear in a model.
    """
    if exclude_names is None:
        exclude_names = []
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name not in exclude_names:
            # Replace
            bit_linear = BitLinear(module.in_features, module.out_features, module.bias is not None)
            bit_linear.weight = module.weight
            bit_linear.bias = module.bias
            setattr(model, name, bit_linear)
        else:
            # Recurse
            replace_linear_with_bitlinear(module, exclude_names)
