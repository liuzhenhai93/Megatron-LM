import torch
import torch.nn.functional as F
import numpy as np
import time
from megatron.model.sparse_attention import XformerSparseAttention as SparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig
from triton.ops.blocksparse import matmul as blocksparse_matmul  # type: ignore
from triton.ops.blocksparse import softmax as blocksparse_softmax  # type: ignore
import math


BATCH = 1
TP = 8
HEADS = 8 // TP
SEQ = 1024
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
    return mask      
    #return torch.ones((1, seq_len, seq_len), dtype=torch.uint8, device = torch.device('cuda:0')) #torch.tril(mask)    




class HandWriteXformerSparse():
    
    def __init__(self,block_size, num_heads, seq_len, causal: bool = True, dropout: float = 0.0):

        config = FixedSparsityConfig(num_heads, attention="unidirectional", block_size=block_size)
        layout = config.make_layout(seq_len)
        layout = torch.tril(layout).to(torch.int32).cuda()    

        #layout = torch.ones([seq_len // block_size, seq_len // block_size]).to(torch.int32).cuda() 

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

        mask = create_mask_for_sparse_attention(self.block_size, self.seq_len)
        self.mask = mask

        def dense_dot_sdd(q, k):
            s = torch.matmul(q, k.permute(*[0, 1, 3, 2]))
            s = s.masked_fill(torch.tril(mask) == 0, 0.0)
            return s
 

        def dense_dot_dsd(p, v):
            return torch.matmul(p, v)

        def dense_softmax(s):
            s = s.masked_fill(torch.tril(mask) == 0, float('-inf'))
            return F.softmax(s, dim=3)

        self.dense_dot_sdd = dense_dot_sdd
        self.dense_dot_dsd = dense_dot_dsd
        self.dense_softmax = dense_softmax

    def forward1(self, q, k, v):
        if not hasattr(self, "sparse_dot_sdd"):
                self.create_triton_kernels(q.device)

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
        if not hasattr(self, "sparse_dot_sdd"):
            self.create_triton_kernels(q.device)

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
        qt = q #/ math.sqrt(float(q.size(-1)))
        s1 = self.sparse_dot_sdd(qt, k)
        s2 = self.dense_dot_sdd(qt, k)
        return s1, s2

    def stod(self, so):
        do = self.sparse_dot_dsd(so, torch.diag(torch.ones([self.seq_len])).unsqueeze(0).unsqueeze(0).expand(BATCH, HEADS, -1, -1).to(torch.float32).cuda())
        #do = do.masked_fill(self.mask == 0, 0.0)
        return do

    def stodv2(self, so):
        data = torch.zeros([1, self.layout.size(0),self.seq_len, self.seq_len]).cuda()
        t = 0
        for i in range(self.layout.size(0)):
            for j in range(self.layout.size(1)):
                for k in range(self.layout.size(2)):
                    if(self.layout[i][j][k] == 1):
                        data[0][i][j*self.block_size:(j+1)*self.block_size, k*self.block_size:(k+1)*self.block_size] = so[0][t][:,:]
                        t = t + 1
        return data                




    def stage2(self, o1, o2):
        return self.sparse_softmax(o1, scale=1.0, is_causal=self.causal), self.dense_softmax(o2)

    def stage3(self, p1, p2, v):
        return self.sparse_dot_dsd(p1, v), self.dense_dot_dsd(p2, v)
      


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


def report_diff(context, a,  b):
    out1 = a.to(torch.float32).detach().cpu().numpy()  
    out2 = b.to(torch.float32).detach().cpu().numpy()
    diff = np.abs(out1 - out2)
    args_max = np.argmax(diff)
    ind = np.unravel_index(args_max, diff.shape)
    print(f"{context}: max_diff={diff[ind]}, out1[{ind}]={out1[ind]} out2[{ind}]={out2[ind]}")
    #print(f"{context}: diff={diff.reshape([-1])}")
    #print(f"{out1} : diff={out1.reshape([-1])}")




def test_stages_precision(block_size):

    attention = HandWriteXformerSparse(block_size, HEADS, SEQ)
    print(f"{attention.block_num*16*16}")

    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
    dtype=torch.float32
    """
    q = torch.randn(*shape, dtype=dtype).cuda()
    k = torch.randn(*shape, dtype=dtype).cuda()
    v = torch.randn(*shape, dtype=dtype).cuda()
    """
    low = 0
    high = 5
    q = torch.randint(low=low, high=high, size=shape).to(dtype).cuda()
    k = torch.randint(low=low, high=high, size=shape).to(dtype).cuda()
    v = torch.randint(low=low, high=high, size=shape).to(dtype).cuda()
    v = torch.ones(size=shape).to(dtype).cuda()*4
    #v = torch.randn(*shape, dtype=dtype).cuda()
    """
    q = torch.ones(size=shape).to(dtype).cuda()*5
    k = torch.ones(size=shape).to(dtype).cuda()*5
    v = torch.ones(size=shape).to(dtype).cuda()*5
    """
    torch.cuda.synchronize()
    
    k.requires_grad_(True)
    q.requires_grad_(True)
    v.requires_grad_(True)
    a = attention.forward1(q, k, v)
    b = attention.forward2(q, k, v)
    torch.cuda.synchronize()
    report_diff(f"block_size{block_size}-attention", a, b)
    stage1_1, stage1_2 = attention.stage1(q, k)
    stage1_1d = attention.stodv2(stage1_1)
    stage1_1d2 = attention.stod(stage1_1)

    print(f"shape {stage1_1.shape}")
    
    print(f"sparse numl {torch.numel(stage1_1)}")
    for i in range(8):
        for j in range(8):
            pass
            #print(f"block {i} {j}")
            #print(stage1_1d[0, 0, i*16:(i+1)*16, j*16:(j+1)*16])


    torch.cuda.synchronize()
    report_diff(f"block_size{block_size}-qk", stage1_1d, stage1_2)
    report_diff(f"block_size{block_size}-stod-stodv2", stage1_1d, stage1_1d2)
      
    stage2_1, stage2_2 = attention.stage2(stage1_1, stage1_1d)
    stage2_1d = attention.stodv2(stage2_1)
    report_diff(f"block_size{block_size}-soft", stage2_1d, stage2_2)
   

    stage3_1, stage3_2 = attention.stage3(stage1_1, stage1_1d, v)
    
    report_diff(f"block_size{block_size}-att", stage3_1, stage3_2)
    
    
if __name__ == "__main__":
    
    for block_size in [16, 32, 64, 128]:
        #test_basic(block_size)    
        #test_precision(block_size)
        #test_stages_precision(block_size)
        pass
    test_stages_precision(16)    
    #print(create_mask_for_sparse_attention(2, 16))
    
