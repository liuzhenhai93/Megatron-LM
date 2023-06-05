

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import math
import numpy as np
import torch

BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, dtype, device, seed=0, scaling=1, struct_mode=False):
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
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = torch.randn((d, d), device=device)
      # col vector is orthogonal
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q, 0, 1)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    if struct_mode:
      torch.random.manual_seed(seed)
      q = create_products_of_givens_rotations(d, seed)
    else:
      torch.random.manual_seed(current_seed)
      unstructured_block = torch.randn((d, d), device=device)
      # col vector is orthogonal
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = torch.vstack(block_list)
  current_seed += 1

  if scaling == 0:
    multiplier = torch.norm(torch.randn((d, d), device=device), dim=1)
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
  del is_query
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
  data_normalizer = 1.0 / data.shape[-1]
  data = data_normalizer * data
  ratio = 1.0 / projection_matrix.shape[0]
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
def causal_nominator(qs, ks, vs):
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
  sums = tf.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])

  result = tf.concat(result, axis=0)
  return result


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix=None):
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
  av_attention = tf.transpose(av_attention, [1, 0, 2, 3]) # [B, L, H , D]
  attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])
  attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
  return av_attention / attention_normalizer


if __name__ == "__main__":
  m = 128
  d = 128
  dtype = torch.float
  device = torch.device("cuda")
  project = create_projection_matrix(m, d, dtype, device)
  #print(torch.matmul(project, torch.transpose(project,0,1)))
   
  # b l h d 
  shape = (1, 256, 8, 128)  
  q = torch.randn(*shape, dtype=dtype).cuda() 
  k = torch.randn(*shape, dtype=dtype).cuda() 
  v = torch.randn(*shape, dtype=dtype).cuda() 

  #q_transformed = softmax_kernel_transformation(q, True, project)
  #print(q.shape)
  #nominator = causal_nominator(q, k, v)[0]
  #print(nominator)

  #[batch_size, length, num_heads, dim_per_head]
  attention = favor_attention(q, k, v, softmax_kernel_transformation, True, project)
  print(attention)





