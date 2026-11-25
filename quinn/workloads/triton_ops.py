"""Trusted diagnostic candidates for fixed float32 unary KernelBench tasks."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def unary(x, y, size: tl.constexpr, block: tl.constexpr, task: tl.constexpr):
    indices = tl.program_id(0) * block + tl.arange(0, block)
    mask = indices < size
    value = tl.load(x + indices, mask=mask, other=0)
    if task == 19:
        value = tl.maximum(value, 0.0)
    elif task == 21:
        value = 1 / (1 + tl.exp(-value))
    else:
        value = libdevice.tanh(value)
    tl.store(y + indices, value, mask=mask)


def candidate(task, block):
    def invoke(x):
        x = x.contiguous()
        result = torch.empty_like(x)
        unary[(triton.cdiv(x.numel(), block),)](x, result, x.numel(), block, task)
        return result
    return invoke
