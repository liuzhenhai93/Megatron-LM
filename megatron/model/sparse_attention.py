import torch

from xformers.components.attention import BlockSparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig

from .module import MegatronModule
from megatron import get_args, core
from megatron.core import mpu

class XformerSparseAttention(MegatronModule):

    def __init__(self, dropout=0.5, causal=True, block_size=None, seq_len=None, num_heads=None):
        super(XformerSparseAttention, self).__init__()
        if (not block_size) or (not seq_len) or (not num_heads):
            args = get_args()

        if not block_size:
            block_size = args.sparse_attn_block_size    

        if not seq_len:
            seq_len = args.seq_length

        if not num_heads:
            world_size = mpu.get_tensor_model_parallel_world_size()
            num_heads = core.utils.divide(args.num_attention_heads, world_size)

        config = FixedSparsityConfig(num_heads, attention="unidirectional",block_size=block_size)
        layout = config.make_layout(seq_len)
        layout = torch.tril(layout).to(torch.int32).cuda() 
        #print(layout)
        #print(torch.nonzero(layout).shape[0]*block_size*block_size)
        self._att = BlockSparseAttention(layout=layout, 
        block_size=block_size, 
        dropout=dropout, 
        num_heads=num_heads, 
        causal=causal)

    def forward(self, q, k, v):
        return self._att(q, k, v)
