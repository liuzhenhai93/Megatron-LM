import torch
import time
from megatron.model.sparse_attention import XformerSparseAttention as SparseAttention

if __name__ == "__main__":
    BATCH = 1
    TP = 8
    HEADS = 96 // TP
    SEQ = 32768
    EMB = 12288 // TP
    BLOCK_SIZE = 16
    DROPOUT = 0.0
    attention = SparseAttention(seq_len = 32768, num_heads = HEADS)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    dtype=torch.bfloat16
    q = torch.randn(*shape, dtype=dtype).cuda()
    q.requires_grad_(True)
    k = torch.randn(*shape, dtype=dtype).cuda()
    k.requires_grad_(True)
    v = torch.randn(*shape, dtype=dtype).cuda()
    v.requires_grad_(True)

    def func(iteration):
        torch.cuda.synchronize()
        start_t = time.time()
        for i in range(iteration): 
            out = attention(q, k, v) 
            out.backward(torch.randn((*out.shape)).to(out.dtype).to("cuda:0"))
        torch.cuda.synchronize()
        end_t = time.time()
        return (end_t - start_t) / iteration

    func(10)
    print(func(10))