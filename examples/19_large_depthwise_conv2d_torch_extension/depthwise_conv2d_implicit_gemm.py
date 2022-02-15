#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.utils.cpp_extension as cpp_extension

CUTLASS_ROOT = os.path.join(os.path.dirname(__file__), "../..")

_extension = None

__all__ = ["DepthWiseConv2dImplicitGEMM"]

def _load_extension():
    global _extension
    if _extension is not None: return _extension
    _extension = cpp_extension.load(
        name="dwconv_implicitgemm",
        sources=[
            "frontend.cpp",
            "forward_fp32.cu",
            "backward_data_fp32.cu",
            "backward_filter_fp32.cu",
        ],
        extra_include_paths=[
            ".",
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "library", "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
            os.path.join(CUTLASS_ROOT, "examples", "common"),
        ],
        verbose=True
    )
    return _extension

class _DepthWiseConv2dImplicitGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x = x.float(); w = w.float()
        ctx.save_for_backward(x, w)
        return _extension.forward_fp32(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        dx = _extension.backward_data_fp32(grad, w)
        dw = _extension.backward_filter_fp32(grad, x, w)
        dx = dx.half(); dw = dw.float()
        return dx, dw

class DepthWiseConv2dImplicitGEMM(nn.Conv2d):
    def __init__(self, channels, kernel, bias=False):
        super().__init__(channels, channels, kernel, groups=channels, bias=bias)
        self._ext = _load_extension()

    def forward(self, x):
        x = _DepthWiseConv2dImplicitGEMM.apply(x, self.weight)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        x = torch.randn(64, 384, 32, 32).cuda()
        m = DepthWiseConv2dImplicitGEMM(384, 31, bias=True).cuda()
        y = m(x)
        print(y.shape)
        y.mean().backward()
        print(m.weight.grad.shape)
