import torch
import time
from megatron.model.sparse_attention import XformerSparseAttention as SparseAttention



BATCH = 1
TP = 8
HEADS = 64 // TP
SEQ = 4*1024
EMB = 8192 // TP
DROPOUT = 0.1


def test(block_size):
    print(f"block size{block_size}")
    attention = SparseAttention(block_size=block_size, seq_len = SEQ, num_heads = HEADS)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    dtype=torch.bfloat16
    q = torch.randn(*shape, dtype=dtype).cuda()
    q.requires_grad_(True)
    k = torch.randn(*shape, dtype=dtype).cuda()
    k.requires_grad_(True)
    v = torch.randn(*shape, dtype=dtype).cuda()
    v.requires_grad_(True)

    def func(iteration):
        out_grad = torch.randn(*shape).to(dtype).to("cuda:0")
        torch.cuda.synchronize()
        start_t = time.time()
        for i in range(iteration): 
            out = attention(q, k, v) 
            #out.backward(out_grad)
        torch.cuda.synchronize()
        end_t = time.time()
        return (end_t - start_t) / iteration

    print(func(10))
    
if __name__ == "__main__":
    for block_size in [16, 32, 64, 128]:
        test(block_size)    