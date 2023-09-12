class FFN():
  def __init__(self, M, d_model, d_ff, voc_size):
    self.M = M
    self.d_model = d_model
    self.d_ff = d_ff
    self.voc_size = voc_size
  def forward(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_ff
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_ff + 9*self.M*self.d_ff + self.M*self.d_model

  def backward(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_ff
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_ff + 13*self.M*self.d_ff

  def weight_update(self, metric):
    if metric == 'macs':
      return 2*self.M*self.d_model*self.d_ff
    elif metric == 'flops':
      return 4*self.M*self.d_model*self.d_ff

    
  def parameters(self):
    W_1 = self.d_model * self.d_ff
    b_1 = self.d_ff
    W_2 = self.d_ff * self.d_model
    b_2 = self.d_model
    return W_1 + b_1 + W_2 + b_2
  
  def activations_bp(self):
    # we count input activations, but we do not count output
    input = self.M * self.d_model
    layer_1 = self.M * self.d_ff
    return input + layer_1
  
  def activations_pep(self):
    # we count input activations, but we do not count output
    input = self.M * self.d_model
    layer_1 = self.M * self.d_ff
    return input + layer_1