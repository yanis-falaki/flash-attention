import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.cpp_extension import load


device = "cuda"

cuda_extension = load(
    name="pybind_name",
    sources=["flash_attn.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class FlashAttention(nn.Module):
    def __init__(self, d, d_q, d_k, d_v):
        super(FlashAttention, self).__init__()
        self.d = d
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.W_query = nn.Parameter(torch.randn(d, d_q))
        self.W_key = torch.nn.Parameter(torch.rand(d, d_k))
        self.W_value = torch.nn.Parameter(torch.rand(d, d_v))

    def forward(self, x):
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value

        context_vector = torch.ops.flash_attn.flash_attn(Q, K, V)
        
        return context_vector


sentence = 'The quick brown fox jumps over a lazy dog'
dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

r = [dc[i] for i in sentence.replace(',', '').split()]
sentence_int = torch.tensor(r, device=device)


d = 3
vocab_size = 50000
torch.manual_seed(0)

embed = nn.Embedding(vocab_size, d, device=device)
embedded_sentence = embed(sentence_int).detach()

fa = FlashAttention(d, d, d, d).to(device)
cv = fa(embedded_sentence)
print(cv.shape)
print(cv)
