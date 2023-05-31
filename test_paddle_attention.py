import paddle
import paddle.nn.functional as F
import paddle.incubate.nn.attn_bias as ab
import numpy as np
import time

from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded,
)
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)


class PaddleAttension(object):
    def __init__(self):
        pass

    def attention(self, q, k, v, causal, method = "naive"):
        if method == "naive":
            return self._naive_attention(q, k, v, causal)
        elif method == "flash":
            return self._flash_attention(q, k, v, causal)
        else:
            return self._memory_efficient_attention(q, k, v, causal)


    def _naive_attention(self, q, k, v, causal):
        # (b, seq, head, hidden) -> (b, head, seq, hidden)
        qt = paddle.transpose(q, [0, 2, 1, 3])
        kt = paddle.transpose(k, [0, 2, 1, 3])
        vt = paddle.transpose(v, [0, 2, 1, 3])
        # scale
        scale = 1.0 / np.sqrt(q.shape[-1])
        # q * k^t, (b, head, seq, hidden), (b, head, hidden, seq)-> (b, head, seq, seq)
        s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
        s = paddle.scale(s, scale)
        # mask or not
        p = (
            paddle.incubate.softmax_mask_fuse_upper_triangle(s)
            if causal
            else F.softmax(s)
        )
        # attention , (b, head, seq, seq) , (b, head, seq, hidden) -> (b, head, seq, hidden)
        o = paddle.matmul(p, vt)
        # (b, seq, head, hidden)
        return paddle.transpose(o, [0, 2, 1, 3])

    def _flash_attention(self, q, k, v, causal):
        out, _ = flash_attention(q, k, v, causal=causal)
        return out

    def _memory_efficient_attention(self, q, k, v, causal):
        scale = 1.0 / np.sqrt(q.shape[-1])
        att_bias = ab.LowerTriangularMask() if causal else None
        out = memory_efficient_attention(
            q,
            k,
            v,
            att_bias,
            0.0,
            scale,
            True
        )
        return out

def time_attention_func(attention, q, k, v, sync=False, iteration=10):
    forward_time = 0.0
    backward_time = 0.0
    for i in range(iteration):
        t = attention(q, k, v, True)
        t.backward()
        paddle.device.cuda.synchronize()
        begin = time.time()
        t2 = attention(q, k, v, True)
        if sync:
            paddle.device.cuda.synchronize()
        end = time.time()
        forward_time += (end - begin)
        t2.backward()
        paddle.device.cuda.synchronize()
        backward_time += (time.time() - begin)
    return forward_time / iteration ,  backward_time / iteration



if __name__ =='__main__':
    paddle_attention = PaddleAttension()
    shape = (1, 32*1024, 12, 128)
    causal = True
    q = np.random.random(shape)
    k = np.random.random(shape)
    v = np.random.random(shape)

    place = paddle.CUDAPlace(0)
    dtype = 'bfloat16'
    q = paddle.to_tensor(
            q, place=place, dtype=dtype, stop_gradient=False
        )
    k = paddle.to_tensor(
            k, place=place, dtype=dtype, stop_gradient=False
        )

    v = paddle.to_tensor(
            v, place=place, dtype=dtype, stop_gradient=False
        )

    def get_attention(method):

        def attention1(q, k, v, causa1):
            return paddle_attention._flash_attention(q, k, v, causal)

        def attention2(q, k, v, causal):
            return paddle_attention._memory_efficient_attention(q, k, v, causal)

        if method == "flash" :
            return attention1
        else:
            return attention2

    for att in ["flash", "mea"]:
        attention = get_attention(att)
        forward_time, backward_time = time_attention_func(attention, q, k, v)
        total_time = forward_time + backward_time
        forward_time, backward_time = time_attention_func(attention, q, k, v, True)
        backward_time_nosync = total_time - forward_time
        print(f"paddle-{att},x,1.0,1.0,{forward_time},{backward_time},{backward_time_nosync},{forward_time+backward_time}, {total_time}")