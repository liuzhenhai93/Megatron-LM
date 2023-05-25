import torch
import torch.nn.functional as F
import numpy as np
import time
from megatron.model.sparse_attention import XformerSparseAttention as SparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig

BATCH = 1
TP = 8
HEADS = 64 // TP
SEQ = 4*1024
EMB = 8192 // TP
DROPOUT = 0.1


def create_mask_for_sparse_attention(block_size, seq_len = SEQ):
    config = FixedSparsityConfig(1, attention="unidirectional", block_size=block_size)
    layout = config.make_layout(seq_len)
    #print(layout)
    mask =  torch.zeros((1, seq_len, seq_len), dtype=torch.uint8, device = torch.device('cuda:0'))
    for i in range (layout.shape[1]):
        for j in range (layout.shape[2]):
            mask[0][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = layout[0][i][j]
    return torch.tril(mask)     



def torch_sparse_attention_simulate(block_size, seq_len = SEQ, num_heads = HEADS):
    # (b, seq, head, hidden) -> (b, seq, head, hidden)
    def attension(q, k, v):
        # (b, seq, head, hidden) -> (b, head, seq, hidden)
        qt = q.permute(*[0, 2, 1, 3])
        kt = k.permute(*[0, 2, 1, 3])
        vt = v.permute(*[0, 2, 1, 3])
        # scale
        scale = 1.0 / np.sqrt(q.shape[-1])
        # q * k^t, (b, head, seq, hidden), (b, head, hidden, seq)-> (b, head, seq, seq)
        s = torch.matmul(qt, kt.permute(*[0, 1, 3, 2]))
        s = s * scale
        mask = create_mask_for_sparse_attention(block_size, seq_len)
        s = s.masked_fill(mask == 0, float('-inf'))
        p = F.softmax(s, dim=3)
        # attension , (b, head, seq, seq) , (b, head, seq, hidden) -> (b, head, seq, hidden)
        o = torch.matmul(p, vt)
        # (b, seq, head, hidden)
        return o.permute(*[0, 2, 1, 3])

    return attension

def xformers_sparse_attension(dropout, block_size, seq_len = SEQ, num_heads = HEADS):
    # (b, head, seq, hidden) -> (b, head, seq, hidden)
    return SparseAttention(dropout=dropout, block_size=block_size, seq_len = seq_len, num_heads = num_heads)



def test_basic(block_size):
    print(f"block size{block_size}")
    attention = xformers_sparse_attension(1.0, block_size=block_size, seq_len = SEQ, num_heads = HEADS)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    dtype=torch.bfloat16
    q = torch.randn(*shape, dtype=dtype).cuda()
    q.requires_grad_(True)
    k = torch.randn(*shape, dtype=dtype).cuda()
    k.requires_grad_(True)
    v = torch.randn(*shape, dtype=dtype).cuda()
    v.requires_grad_(True)

    def func(iteration):
        out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
        torch.cuda.synchronize()
        start_t = time.time()
        for i in range(iteration): 
            out = attention(q, k, v) 
            out.backward(out_grad)
        torch.cuda.synchronize()
        end_t = time.time()
        return (end_t - start_t) / iteration

    print(f"block size {block_size} {func(10)}")

def test_precision(block_size):
    torch_dense_simulator = torch_sparse_attention_simulate(block_size=block_size, seq_len = SEQ, num_heads = HEADS)
    sparse_attention = xformers_sparse_attension(0, block_size=block_size, seq_len = SEQ, num_heads = HEADS)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    dtype=torch.float32
    q = torch.randn(*shape, dtype=dtype).cuda()
    q.requires_grad_(True)
    k = torch.randn(*shape, dtype=dtype).cuda()
    k.requires_grad_(True)
    v = torch.randn(*shape, dtype=dtype).cuda()
    v.requires_grad_(True)
    

    def func1():
        out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
        torch.cuda.synchronize()
        # (b, h, s, d) -> (b, s, h, d)
        qt = q.permute(*[0, 2, 1, 3])
        kt = k.permute(*[0, 2, 1, 3])
        vt = v.permute(*[0, 2, 1, 3])
        out = torch_dense_simulator(qt, kt, vt)
        out = out.permute(*[0, 2, 1, 3])
        torch.cuda.synchronize()
        return out

    def func2():
        out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
        torch.cuda.synchronize()
        out = sparse_attention(q, k, v)
        torch.cuda.synchronize()
        return out

    out1 = func1()
    out2 = func2() 
    out1 = torch.reshape(out1.to(torch.float32).detach(), [-1]).cpu().numpy()  
    out2 = torch.reshape(out2.to(torch.float32).detach(), [-1]).cpu().numpy()
    diff = out1 - out2
    print("block_size {} min diff {}; max diff {}".format(block_size, np.min(diff), np.max(diff)) )
    args_min = np.argmin(diff)
    print(f"block_size {block_size} arg_min {args_min}, out1 {out1[args_min]} out2 {out2[args_min]}")
    args_max = np.argmax(diff)
    print(f"block_size {block_size} arg_max {args_max}, out1 {out1[args_max]} out2 {out2[args_max]}")
    print(f"block_size {block_size} diff {diff}")

    



    
if __name__ == "__main__":
    
    for block_size in [16, 32, 64, 128]:
        #test_basic(block_size)    
        test_precision(block_size)
    
