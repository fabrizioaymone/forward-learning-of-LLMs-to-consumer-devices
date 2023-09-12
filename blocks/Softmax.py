class Softmax():
  def __init__(self, M, d_model, d_ff, voc_size):
    self.M = M
    self.d_model = d_model
    self.d_ff = d_ff
    self.voc_size = voc_size
  def forward(self,metric):
    if metric == 'macs':
      return self.M*self.d_model*self.voc_size
    elif metric == 'flops':
      return 2*self.M*self.d_model*self.voc_size + 5*self.M*self.voc_size

  def backward(self, metric):
    if metric == 'macs':
      return self.M*self.voc_size*self.d_model
    elif metric == 'flops':
      return 2*self.M*self.voc_size*self.d_model

  def weight_update(self,metric):
    if metric == 'macs':
      return self.M*self.voc_size*self.d_model
    elif metric == 'flops':
      return 2*self.M*self.voc_size*self.d_model

  def parameters(self):
    W_s = self.d_model * self.voc_size
    return W_s
  
  def activations_bp(self):
    # we do count input activations and also output activations
    input = self.M * self.d_model
    soft = self.M * self.d_model
    return input + soft
  
  def activations_pep(self):
    # we do count input activations and also output activations
    input = self.M * self.d_model
    soft = self.M * self.d_model
    return input + soft