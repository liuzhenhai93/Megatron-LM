

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import time

BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, dtype, device, seed=0, scaling=0, struct_mode=True):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt
      orthogonalization.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = int(m / d)
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    if struct_mode:
      q = create_products_of_givens_rotations(d, current_seed).to(device).to(dtype)
    else:
      torch.random.manual_seed(current_seed)
      torch.cuda.manual_seed(current_seed)
      unstructured_block = torch.randn((d, d), dtype=torch.float64, device=device)
      # col vector is orthogonal
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q.to(dtype), 0, 1)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    if struct_mode:
      q = create_products_of_givens_rotations(d, current_seed).to(device).to(dtype)
    else:
      torch.random.manual_seed(current_seed)
      torch.cuda.manual_seed(current_seed)
      unstructured_block = torch.randn((d, d), device=device)
      # col vector is orthogonal
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = torch.vstack(block_list)
  current_seed += 1

  if scaling == 0:
    multiplier = torch.norm(torch.randn((m, d), device=device), dim=1)
  elif scaling == 1:
    multiplier = np.sqrt(float(d)) * torch.ones((m)).to(device)
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return torch.matmul(torch.diag(multiplier), final_matrix).to(dtype)


def create_products_of_givens_rotations(dim, seed):
  r"""Constructs a 2D-tensor which is a product of Givens random rotations.

  Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
  random rotation. The resulting tensor mimics a matrix taken uniformly at
  random from the orthogonal group.

  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.

  Returns:
    The product of Givens random rotations.
  """
  nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
  q = np.eye(dim, dim)
  np.random.seed(seed)
  for _ in range(nb_givens_rotations):
    random_angle = math.pi * np.random.uniform()
    random_indices = np.random.choice(dim, 2)
    index_i = min(random_indices[0], random_indices[1])
    index_j = max(random_indices[0], random_indices[1])
    slice_i = q[index_i]
    slice_j = q[index_j]
    new_slice_i = math.cos(random_angle) * slice_i + math.sin(
        random_angle) * slice_j
    new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
        random_angle) * slice_j
    q[index_i] = new_slice_i
    q[index_j] = new_slice_j
  return torch.from_numpy(q)  

def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  if projection_matrix is None:
    return tf.nn.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  data_normalizer = 1.0 
  if not is_query:
    data_normalizer = 1.0 / np.sqrt(np.sqrt(float(data.shape[-1])))
  data = data_normalizer * data
  ratio = 1.0 / np.sqrt(float(projection_matrix.shape[0]))
  data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = torch.square(data)
  diag_data = torch.sum(diag_data, dim=data.dim() - 1)
  diag_data = diag_data / 2.0
  diag_data = torch.unsqueeze(diag_data, dim=data.dim() - 1)
  last_dims_t = len(data_dash.shape) - 1
  attention_dims_t = len(data_dash.shape) - 3
  max_value = torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
  if not is_query:
    max_value = torch.max(max_value, dim=attention_dims_t, keepdim=True)[0]
  data_dash = ratio * (torch.exp(data_dash - diag_data - max_value) + numerical_stabilizer)
  return data_dash


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
  return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  all_ones = torch.ones([ks.shape[0]])
  ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
  return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


#@tf.custom_gradient
def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])
  result = torch.concat(result, dim=0)
  return result


#@tf.custom_gradient
def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = torch.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(torch.sum(qs[index] * sums, dim=2)[None, Ellipsis])

  result = torch.concat(result, dim=0)
  return result


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix=softmax_kernel_transformation):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  query_prime = torch.permute(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
  key_prime = torch.permute(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
  value = torch.permute(value, [1, 0, 2, 3])  # [L,B,H,D]

  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  av_attention = torch.permute(av_attention, [1, 0, 2, 3]) # [B, L, H , D]
  attention_normalizer = torch.permute(attention_normalizer, [1, 0, 2])
  attention_normalizer = torch.unsqueeze(attention_normalizer, dim=len(attention_normalizer.shape))
  return av_attention / attention_normalizer


def naive_attention(q, k, v, causal):
  # (b, seq, head, hidden) -> (b, head, seq, hidden)
  qt = q.permute(*[0, 2, 1, 3])
  kt = k.permute(*[0, 2, 1, 3])
  vt = v.permute(*[0, 2, 1, 3])
  # scale
  scale = 1.0 / np.sqrt(q.shape[-1])
  # q * k^t, (b, head, seq, hidden), (b, head, hidden, seq)-> (b, head, seq, seq)
  s = torch.matmul(qt, kt.permute(*[0, 1, 3, 2]))
  s = s * scale
  # mask or not
  if causal:
      seq_len = s.shape[-1]
      mask = torch.triu(torch.ones((1, seq_len, seq_len),dtype=torch.uint8, device = torch.device('cuda:0')), diagonal=1)
      s = s.masked_fill(mask == 1, float('-inf'))
  p = F.softmax(s, dim=3)
  # attension , (b, head, seq, seq) , (b, head, seq, hidden) -> (b, head, seq, hidden)
  o = torch.matmul(p, vt)
  # (b, seq, head, hidden)
  return o.permute(*[0, 2, 1, 3])


def report_diff(context, a,  b):
  out1 = a.to(torch.float32).detach().cpu().numpy().flatten()  
  out2 = b.to(torch.float32).detach().cpu().numpy().flatten()
  diff = np.abs(out1 - out2)
  args_max = np.argmax(diff)
  ind = np.unravel_index(args_max, diff.shape)
  mean = np.mean(diff)
  var = np.var(diff)
  print(f"{context}: max_diff={diff[ind]}, out1[{ind}]={out1[ind]} out2[{ind}]={out2[ind]}, diff mean={mean} var={var}")  


def time_func(func, iteration=10):
    time_interval = 0.0
    for i in range(iteration):
        func()
        torch.cuda.synchronize()
        begin = time.time()
        func()
        end = time.time()
        torch.cuda.synchronize()
        time_interval += (end - begin)
    return time_interval / iteration


if __name__ == "__main__":
  m = 128
  d = 16
  dtype = torch.float
  device = torch.device("cuda")
  project = create_projection_matrix(m, d, dtype, device)
  print(torch.matmul(project, torch.transpose(project,0,1)))
   
  # b l h d 
  shape = (1, 4096, 8, 16)  
  seed = 2
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  q = torch.rand(*shape, dtype=dtype).cuda() 
  k = torch.rand(*shape, dtype=dtype).cuda() 
  v = torch.rand(*shape, dtype=dtype).cuda() 

  #q_transformed = softmax_kernel_transformation(q, True, project)
  #print(q.shape)
  #nominator = causal_nominator(q, k, v)[0]
  #print(nominator)
  #[batch_size, length, num_heads, dim_per_head]

  _ = favor_attention(q, k, v, softmax_kernel_transformation, True, project)
  torch.cuda.synchronize()
  time_begin = time.time()
  favor_result = favor_attention(q, k, v, softmax_kernel_transformation, True, project)
  torch.cuda.synchronize()

  #print(attention)

  naive_result = naive_attention(q, k, v, True)

  report_diff("favor vs naive", favor_result, naive_result)
  #print(favor_result)
  def func1():
    return favor_attention(q, k, v, softmax_kernel_transformation, True, project)

  def func2():
    return naive_attention(q, k, v, True)

  print(time_func(func1))
  print(time_func(func2))



  





