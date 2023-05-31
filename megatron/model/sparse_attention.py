import torch

from xformers.components.attention import BlockSparseAttention
from xformers.components.attention.sparsity_config import FixedSparsityConfig
from xformers.components.attention.sparsity_config import BigBirdSparsityConfig

from .module import MegatronModule
from megatron import get_args, core
from megatron.core import mpu


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

class XformerSparseAttention(MegatronModule):

    def __init__(self, dropout=0.5, causal=True, sparse_type=None, block_size=None, seq_len=None, num_heads=None):
        super(XformerSparseAttention, self).__init__()
        if (not block_size) or (not seq_len) or (not num_heads) or (not sparse_type):
            args = get_args()

        if not block_size:
            block_size = args.sparse_attn_block_size    

        if not seq_len:
            seq_len = args.seq_length

        if not num_heads:
            world_size = mpu.get_tensor_model_parallel_world_size()
            num_heads = core.utils.divide(args.num_attention_heads, world_size)

        if not sparse_type:
            sparse_type = args.sparse_attn_type

        layout = create_layout(sparse_type, num_heads, seq_len, block_size, causal)    
        self._att = BlockSparseAttention(layout=layout, 
        block_size=block_size, 
        dropout=dropout, 
        num_heads=num_heads, 
        causal=causal)

    def forward(self, q, k, v):
        return self._att(q, k, v)
