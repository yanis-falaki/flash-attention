import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

cuda_extension = load(
    name="pybind_name",
    sources=["vector_add_cuda.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

x = torch.randn(3, device="cuda")
y = torch.randn(3, device="cuda")
z = torch.ops.cuda_extension.vector_add(x, y)

print(z == x + y)