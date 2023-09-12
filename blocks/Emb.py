class Emb():
  # it considers "dense" embedding lookups (i.e., multiplication by a one-hot vector) as in electra
  def __init__(self, M, d_model, voc_size):
    self.M = M
    self.d_model = d_model
    self.voc_size = voc_size
  def forward(self, metric):
    if metric == 'macs':
      return self.M * self.voc_size * self.d_model
    elif metric == 'flops':
      return  2*self.M * self.voc_size * self.d_model