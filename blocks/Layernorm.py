class LayerNorm():
  def __init__(self, M, d_model, d_ff, voc_size):
    self.M = M
    self.d_model = d_model
    self.d_ff = d_ff
    self.voc_size = voc_size

  def forward(self, metric):
    if metric == 'macs':
      return 0
    elif metric == 'flops':
      return 9*self.M*self.d_model

  def backward(self, metric):
    if metric == 'macs':
      return self.M*self.d_model*self.d_model
    elif metric == 'flops':
      return 11*self.M*self.d_model*self.d_model + 2*self.M*self.d_model

  def weight_update(self, metric):
    if metric == 'macs':
      return self.M*self.d_model
    elif metric == 'flops':
      return 3*self.M*self.d_model

    
  def parameters(self):
    gamma = self.d_model
    beta = self.d_model
    return gamma + beta
  
  def activations_bp(self):
    # we do count input activations, but we do not count output
    input = self.M * self.d_model # we need it to perform the derivative of the normalized w.r.t the input
    normalized_before_scaling = self.M * self.d_model
    return input + normalized_before_scaling
  
  def activations_pep(self):
    # we do count input activations, but we do not count output
    input = self.M * self.d_model # we need it to perform the derivative of the normalized w.r.t the input
    normalized_before_scaling = self.M * self.d_model
    return input + normalized_before_scaling