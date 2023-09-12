class Attn():
  def __init__(self, M, N, d_model, voc_size, h):
    self.M = M
    self.N = N
    self.d_model = d_model
    self.voc_size = voc_size
    self.h = h
  def forward(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_model + 2*self.M*self.N*self.d_model + 2*self.N*self.d_model*self.d_model
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_model + 4*self.M*self.N*self.d_model + 4*self.N*self.d_model*self.d_model + 6*self.M*self.N*self.h

  def backward(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_model +  2*self.N*self.d_model*self.d_model + 4*self.M*self.N*self.d_model + self.M * self.N * self.M * self.h
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_model +  4*self.N*self.d_model*self.d_model + 8*self.M*self.N*self.d_model + 2 * self.M * self.N * self.M * self.h + self.M*self.N*self.h

  def weight_update(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_model +  2*self.N*self.d_model*self.d_model
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_model +  4*self.N*self.d_model*self.d_model

  def parameters(self):
    W_q = self.d_model * self.d_model
    W_k = self.d_model * self.d_model
    W_v = self.d_model * self.d_model
    W_o = self.d_model * self.d_model
    return W_q + W_k + W_v + W_o
  
  def activations_bp(self):
    # we count input activations, but we do not count output
    input = self.M * self.d_model
    Q = self.M * self.d_model
    K = self.N * self.d_model
    V = self.N * self.d_model
    soft = self.M * self.N
    pre_W_o = self.M * self.d_model
    return input + Q + K + V + soft + pre_W_o
  
  def activations_pep(self):
    # we count input activations, but we do not count output
    # soft activations not needed w.r.t bakcpropo
    input = self.M * self.d_model
    Q = self.M * self.d_model
    K = self.N * self.d_model
    V = self.N * self.d_model
    #soft = self.M * self.N
    pre_W_o = self.M * self.d_model
    return input + Q + K + V + pre_W_o

