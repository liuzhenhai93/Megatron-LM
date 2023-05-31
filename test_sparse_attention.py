import math
import contextlib
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from triton.ops.blocksparse import matmul as blocksparse_matmul 
from triton.ops.blocksparse import softmax as blocksparse_softmax  
from xformers.components.attention import BlockSparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig
from xformers.components.attention.sparsity_config import BigBirdSparsityConfig
from triton.ops import matmul as dense_matmul

try: 
    from megatron.model.transformer import FlashSelfAttention
except ImportError:
    FlashSelfAttention = None  


BATCH = 1
TP = 8
HEADS = 1
SEQ = 1024*32
EMB = 128*HEADS
DROPOUT = 0.1


def create_layout(sparse_type, num_heads, seq_len, block_size, causal=True, **kwargs):
    assert sparse_type.upper() in ["BIRD", "FIX"], "type must be BIRD, or FIX"

    def fix_layout():
        config = FixedSparsityConfig(num_heads, attention="unidirectional",block_size=block_size, **kwargs)
        layout = config.make_layout(seq_len)
        return layout

    def bird_layout():
        config = BigBirdSparsityConfig(num_heads, attention="bidirectional",block_size=block_size, **kwargs)
        layout = config.make_layout(seq_len)
        return layout
    #print(f"seq_len {seq_len} num_heads {num_heads}")    
    layout = fix_layout() if sparse_type.upper() == "FIX" else bird_layout()
    layout = torch.tril(layout) if causal else layout
    layout = layout.to(torch.int32).cuda()  
    return  layout


def create_mask_for_sparse_attention(block_size, seq_len = SEQ, sparse_type="fix",**kwargs):
    layout = create_layout(sparse_type=sparse_type,num_heads=1, seq_len=seq_len, block_size=block_size, causal=True, **kwargs)
    mask =  torch.zeros((1, seq_len, seq_len), dtype=torch.uint8, device = torch.device('cuda:0'))
    for i in range (layout.shape[1]):
        for j in range (layout.shape[2]):
            mask[0][i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = layout[0][i][j]  
    return mask      

def create_sparse_attention(sparse_type, block_size, seq_len, num_heads=1, dropout=0, causal=True,**kwargs):
    layout = create_layout(sparse_type=sparse_type,num_heads=num_heads, seq_len=seq_len, 
    block_size=block_size, causal=True, **kwargs)
    return BlockSparseAttention(layout=layout, block_size=block_size, 
    dropout=dropout, num_heads=num_heads, causal=causal)

def time_attention_func(attention, q, k, v, out_grad, sync=False, iteration=10):
    forward_time = 0.0
    backward_time = 0.0
    for i in range(iteration):
        t = attention(q, k, v)
        t.backward(out_grad)
        torch.cuda.synchronize()
        begin = time.time()
        t2 = attention(q, k, v)
        if sync:
            torch.cuda.synchronize()
        end = time.time()
        forward_time += (end - begin)
        t2.backward(out_grad)
        torch.cuda.synchronize()
        backward_time += (time.time() - begin)
    return forward_time / iteration ,  backward_time / iteration    


def test_flash_attention_performance(dtype = torch.bfloat16):
    if not FlashSelfAttention:
        return
    attention = FlashSelfAttention(causal=True)
    shape = (BATCH, SEQ, HEADS, EMB // HEADS)  
    q = torch.randn(*shape, dtype=dtype).cuda() 
    k = torch.randn(*shape, dtype=dtype).cuda() 
    v = torch.randn(*shape, dtype=dtype).cuda() 
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
    forward_time, backward_time = time_attention_func(attention, q, k, v, out_grad)
    total_time = forward_time + backward_time
    forward_time, backward_time = time_attention_func(attention, q, k, v, out_grad, True)
    backward_time_nosync = total_time - forward_time
    print(f"flash-attention,x,1.0,1.0,{forward_time},{backward_time},{backward_time_nosync},{forward_time+backward_time}, {total_time}")
    
          

def test_sparse_attention_performance(block_size, dtype, sparse_type="fix",**kwargs):
    attention = create_sparse_attention(sparse_type, block_size=block_size, seq_len=SEQ, num_heads=HEADS, dropout=0,causal=True,**kwargs)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    q = torch.randn(*shape, dtype=dtype).cuda() 
    k = torch.randn(*shape, dtype=dtype).cuda() 
    v = torch.randn(*shape, dtype=dtype).cuda() 
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
    forward_time, backward_time = time_attention_func(attention, q, k, v, out_grad)
    total_time = forward_time + backward_time
    forward_time, backward_time = time_attention_func(attention, q, k, v, out_grad, True)
    backward_time_nosync = total_time - forward_time

    layout = create_layout(sparse_type=sparse_type,num_heads=1, seq_len=SEQ, block_size=block_size, causal=True, **kwargs)
    non_zero_block_num = torch.sum(layout)
    block = SEQ // block_size
    total =  block * block
    c_ratio = non_zero_block_num / total
    a_ratio = c_ratio - ((block_size - 1)/(2*block_size))*(1/(block))
    print(f'{sparse_type},"{kwargs}",{2*c_ratio},{2*a_ratio},{forward_time},{backward_time},{backward_time_nosync},{forward_time+backward_time},{total_time}')
    


class HandWriteXformerSparse():
    
    def __init__(self,block_size, num_heads, seq_len, sparse_type="fix", causal: bool = True, dropout: float = 0.0):

        layout = create_layout(sparse_type=sparse_type,num_heads=num_heads, 
        seq_len=seq_len, block_size=block_size, causal=causal)

        assert block_size in (
                16,
                32,
                64,
                128,
            ), "Only block sizes in [16, 32, 64, 128] are supported"

        if layout.dim() == 2:
                print(
                    "The layout passed is lacking a head dimension and a batch dimension"
                )
                print(
                    "Now assuming that the same layout is to be used across all heads"
                )
                layout = layout.unsqueeze(0).expand(num_heads, -1, -1)    

        self.block_size = block_size
        self.layout = layout
        self.dropout = dropout
        self.causal = causal
        self.seq_len = seq_len
        self.block_num = torch.sum(self.layout)

        mask = create_mask_for_sparse_attention(self.block_size, self.seq_len, sparse_type=sparse_type)
        self.mask = mask
        

    def create_triton_kernels(self, device):
        # blocksparse operators

         # (b, head, seq, hidden)  (b, head, seq, hidden) ->  (b, head, seq, seq)
        self.sparse_dot_sdd = blocksparse_matmul(
            self.layout,
            self.block_size,
            "sdd",
            trans_a=False,
            trans_b=True,
            device=device,
        )

        # (b, head, seq, seq) * (b, head, seq, hidden) -> (b, head, seq, hidden)
        self.sparse_dot_dsd = blocksparse_matmul(
                self.layout,
                self.block_size,
                "dsd",
                trans_a=False,
                trans_b=False,
                device=device,
        )
        
        # (b, head, seq, seq)
        self.sparse_softmax = blocksparse_softmax(
            self.layout,
            self.block_size,
            device=device,
        )

        def dense_dot_sdd(q, k):
            s = torch.matmul(q, k.permute(*[0, 1, 3, 2]))
            s = s.masked_fill(self.mask == 0, 0.0)
            return s
 

        def dense_dot_dsd(p, v):
            return torch.matmul(p, v)

        def dense_softmax(s):
            s = s.masked_fill(torch.tril(self.mask) == 0, float('-inf'))
            return F.softmax(s, dim=3)

        self.dense_dot_sdd = dense_dot_sdd
        self.dense_dot_dsd = dense_dot_dsd
        self.dense_softmax = dense_softmax

    def try_create_kernel(self, device):
        if not hasattr(self, "sparse_dot_sdd"):
                self.create_triton_kernels(device)


    def forward1(self, q, k, v):
        self.try_create_kernel(q.device)
        assert (
                q.shape[-2] == k.shape[-2]
        ), "Blocksparse requires the same dimensions for K and Q for now"

        assert (
                q.shape[-2] == self.layout.shape[-2] * self.block_size
        ), "Actual sequence size and layout are inconsistent"
        assert (
                k.shape[-2] == self.layout.shape[-2] * self.block_size
        ), "Actual sequence size and layout are inconsistent"

        assert (
                q.shape[-2] % self.block_size) == 0, "Sequence length {}  must be a multiple of block size {}".format(
                q.shape[-2], self.block_size)

        q = q / math.sqrt(q.size(-1))
        sparse_att_mat = self.sparse_dot_sdd(q, k)
        # - softmax on the sparse attention matrix
        sparse_att_mat = self.sparse_softmax(sparse_att_mat, scale=1.0, is_causal=self.causal)
        # sparse_att_mat = self.attn_drop(sparse_att_mat)
        # - then (dense) attention is (sparse) attention matrix * dense (value)
        a = self.sparse_dot_dsd(sparse_att_mat, v)
        return a   


    def forward2(self, q, k, v):
        self.try_create_kernel(q.device)
        assert (
                q.shape[-2] == k.shape[-2]
        ), "Blocksparse requires the same dimensions for K and Q for now"

        assert (
                q.shape[-2] == self.layout.shape[-2] * self.block_size
        ), "Actual sequence size and layout are inconsistent"
        assert (
                k.shape[-2] == self.layout.shape[-2] * self.block_size
        ), "Actual sequence size and layout are inconsistent"

        assert (
                q.shape[-2] % self.block_size) == 0, "Sequence length {}  must be a multiple of block size {}".format(
                q.shape[-2], self.block_size)

        q = q / math.sqrt(q.size(-1))
        sparse_att_mat = self.dense_dot_sdd(q, k)
        # - softmax on the sparse attention matrix
        sparse_att_mat = self.dense_softmax(sparse_att_mat)
        #sparse_att_mat = self.attn_drop(sparse_att_mat)
        # - then (dense) attention is (sparse) attention matrix * dense (value)
        a = self.dense_dot_dsd(sparse_att_mat, v)
        return a   

    def stage1(self, q, k):
        self.try_create_kernel(q.device)
        qt = q / math.sqrt(float(q.size(-1)))
        s1 = self.sparse_dot_sdd(qt, k)
        s2 = self.dense_dot_sdd(qt, k)
        return s1, s2

    def stod(self, so, dtype=torch.float32):
        self.try_create_kernel(so.device)
        do = self.sparse_dot_dsd(so, torch.diag(torch.ones([self.seq_len])).unsqueeze(0).unsqueeze(0).expand(BATCH, HEADS, -1, -1).to(dtype).cuda())
        #do = do.masked_fill(self.mask == 0, 0.0)
        return do

    def stage2(self, o1, o2):
        self.try_create_kernel(o1.device)
        return self.sparse_softmax(o1, scale=1.0, is_causal=self.causal), self.dense_softmax(o2)

    def stage3(self, p1, p2, v):
        self.try_create_kernel(p1.device)
        return self.sparse_dot_dsd(p1, v), self.dense_dot_dsd(p2, v)




def report_diff(context, a,  b):
    out1 = a.to(torch.float32).detach().cpu().numpy()  
    out2 = b.to(torch.float32).detach().cpu().numpy()
    diff = np.abs(out1 - out2)
    args_max = np.argmax(diff)
    ind = np.unravel_index(args_max, diff.shape)
    print(f"{context}: max_diff={diff[ind]}, out1[{ind}]={out1[ind]} out2[{ind}]={out2[ind]}")
    #print(f"{context}: diff={diff.reshape([-1])}")
    #print(f"{out1} : diff={out1.reshape([-1])}")




def test_stages_precision(block_size, dtype=torch.bfloat16, sparse_type="fix"):
    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    attention = HandWriteXformerSparse(block_size, HEADS, SEQ, sparse_type=sparse_type)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    q = torch.randn(*shape, dtype=dtype).cuda() 
    k = torch.randn(*shape, dtype=dtype).cuda() 
    v = torch.randn(*shape, dtype=dtype).cuda() 

    q1 = q.detach()
    k1 = k.detach()
    v1 = v.detach()

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    q1.requires_grad_(True)
    k1.requires_grad_(True)
    v1.requires_grad_(True)
    
    torch.cuda.synchronize()
    
    out_grad = torch.ones(*shape).to(dtype).to("cuda:0")
    
    a = attention.forward1(q, k, v)
    b = attention.forward2(q1, k1, v1)
    a.backward(out_grad)
    b.backward(out_grad)
    torch.cuda.synchronize()
    #report_diff(f"sparse_type {sparse_type} block_size[{block_size}]-dtype[{dtype}]-attention", a, b)
    #report_diff(f"sparse_type {sparse_type} block_size[{block_size}]-dtype[{dtype}]-q_grad", q.grad, q1.grad)
    #report_diff(f"sparse_type {sparse_type} block_size[{block_size}]-dtype[{dtype}]-k_grad", k.grad, k1.grad)
    #report_diff(f"sparse_type {sparse_type} block_size[{block_size}]-dtype[{dtype}]-v_grad", v.grad, v1.grad)


    """
    stage1_1, stage1_2 = attention.stage1(q, k)
    stage1_1d = attention.stod(stage1_1, dtype)
    torch.cuda.synchronize()
    report_diff(f"block_size{block_size}-qk", stage1_1d, stage1_2)

    stage2_1, stage2_2 = attention.stage2(stage1_1, stage1_1d)
    torch.cuda.synchronize()
    stage2_1d = attention.stodv2(stage2_1)
    torch.cuda.synchronize()
    report_diff(f"block_size{block_size}-soft", stage2_1d, stage2_2)

    torch.cuda.synchronize()
    stage3_1, stage3_2 = attention.stage3(stage1_1, stage1_1d, v)
    torch.cuda.synchronize()
    report_diff(f"block_size{block_size}-att", stage3_1, stage3_2)
    """

def print_sparsety(seq_len, block_size, dtype=torch.bfloat16, sparse_type="fix", **kwargs):
    layout = create_layout(sparse_type=sparse_type,num_heads=1, seq_len=seq_len, block_size=block_size, causal=True, **kwargs)
    non_zero_block_num = torch.sum(layout)
    block = seq_len // block_size
    total =  block * block
    c_ratio = non_zero_block_num / total
    a_ratio = c_ratio - ((block_size - 1)/(2*block_size))*(1/(block))
    print(f"{sparse_type} {kwargs} block_size {block_size}  block ={non_zero_block_num}| total {total} comp load {c_ratio}| algri load {a_ratio}")


def test_performance():
    # base line flash attension
    print("attr-type,config,comp-load,algri-load,forward,backward,backward_nosync,total, total_nosync")
    test_flash_attention_performance()
    #print("| --- | ---- | ---- | ---- | ---- | ---- | ---- |")
    block_size = 64
    """
    sparse_type = "fix"
    for dtype in [torch.bfloat16]:
        for (num_local_blocks, num_global_blocks) in [(4, 1),(8, 1),(8, 2),(16, 2),(16,4), (32, 2), (32, 4), (32, 8), (64, 4), (64, 8)]:
            test_sparse_attention_performance(block_size, dtype, sparse_type, num_local_blocks=num_local_blocks, num_global_blocks=num_global_blocks)   
    """
    sparse_type = "bird"
    """
    for dtype in [torch.bfloat16]:
        for i in range(8):
            j = 2**i
            num_sliding_window_blocks = 2*j + 1 
            num_global_blocks = j
            num_random_blocks = j
            test_sparse_attention_performance(block_size, dtype, sparse_type, num_sliding_window_blocks=num_sliding_window_blocks, num_global_blocks=num_global_blocks, num_random_blocks=num_random_blocks)   
    """
   
    for dtype in [torch.bfloat16]:
        for i in range(16,64,2):
            j = i #32**i
            num_sliding_window_blocks = 2*j + 1 
            num_random_blocks = 2*j + 1
            num_global_blocks = 2*j
            test_sparse_attention_performance(block_size, dtype, sparse_type, num_sliding_window_blocks=num_sliding_window_blocks, num_global_blocks=num_global_blocks, num_random_blocks=num_random_blocks)   


if __name__ == "__main__":
    test_performance()  
    print_sparsety(seq_len=4096, block_size=64, sparse_type="bird", num_sliding_window_blocks=3, num_global_blocks=2, num_random_blocks=3)

        
