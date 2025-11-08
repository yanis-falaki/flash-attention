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

class Attentions(nn.Module):
    def __init__(self, num_heads, d):
        super(Attentions, self).__init__()
        self.d = d
        self.d_q = d
        self.d_k = d
        self.d_v = d

        self.W_query = nn.Parameter(torch.randn(1, num_heads, d, self.d_q))
        self.W_key = torch.nn.Parameter(torch.rand(1, num_heads, d, self.d_k))
        self.W_value = torch.nn.Parameter(torch.rand(1, num_heads, d, self.d_v))

    def forward(self, x):
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value

        attention_scores_torch = Q @ K.transpose(2, 3) / math.sqrt(self.d_k)
        attention_weights_torch = F.softmax(attention_scores_torch, dim=3)
        context_vector_torch = attention_weights_torch @ V

        context_vector_flash = torch.ops.flash_attn.flash_attn(Q, K, V)
        
        return context_vector_torch, context_vector_flash


sentence = 'The quick brown fox jumps over a lazy dog'
dc = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)

r = [dc[i] for i in sentence.replace(',', '').split()]
sentence_int = torch.tensor(r, device=device)


d = 3
vocab_size = 50000
torch.manual_seed(0)

embed = nn.Embedding(vocab_size, d, device=device)
embedded_sentence = embed(sentence_int).unsqueeze(0).detach()

ats = Attentions(1, d).to(device)
cvs = ats(embedded_sentence)
print(torch.abs(cvs[0] - cvs[1]))
