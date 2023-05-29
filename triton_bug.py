import torch
from triton.ops import matmul as triton_matmul

s=torch.diag(torch.ones([8])).to(torch.float32).cuda()
v=torch.diag(torch.ones([8])).to(torch.float32).cuda()*2875.0
print(f"s:{s}")
print(f"v:{v}")
torch_sv = torch.matmul(s, v)
triton_sv = triton_matmul(s, v)
print(f"torch s v matmul: {torch_sv}")
print(f"triton s v matmul: {triton_sv}")
torch.cuda.synchronize()
