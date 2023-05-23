from xformers.components.attention import BlockSparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig

from .module import MegatronModule
from megatron import get_args, core
from megatron.core import mpu

class XformerSparseAttention(MegatronModule):

    def __init__(self, block_size=16, dropout=0.5, causal = True, seq_len=None, num_heads=None):
         args = get_args()
        if not seq_len:
            seq_len = args.seq_length

        if not num_heads:
            world_size = mpu.get_tensor_model_parallel_world_size()
            num_heads = core.utils.divide(args.num_attention_heads, world_size)

        config = FixedSparsityConfig(num_heads, block_size=block_size)
        layout = config.make_layout(seq_len)
        layout = torch.tril(layout).to(torch.int32).cuda() 
        self._att = BlockSparseAttention(layout=layout, 
        block_size=block_size, 
        dropout=dropout, 
        num_heads=num_heads, 
        causal=causal)

    def forward(self, q, k, v):
        return self._att(q, k, v)


if __name__ == "__main__":
    BATCH = 1
    TP = 8
    HEADS = 96 // TP
    SEQ = 32768
    EMB = 12288 // TP
    BLOCK_SIZE = 16
    DROPOUT = 0.0
    print(blocks)
    attention = XformerSparseAttention(seq_len = 32768, num_heads = HEADS)
    shape = (BATCH, HEADS, SEQ, EMB // HEADS)  
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