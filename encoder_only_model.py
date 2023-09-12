from blocks.Attn import *
from blocks.FFN import *
from blocks.Layernorm import *
from blocks.Emb import *
from blocks.Softmax import *

from tabulate import tabulate



class encoder_only_model():
  def __init__(self,n_dec, M, MASKED, d_model, d_ff, h, voc_size):
    self.n_dec = n_dec
    self.M = M
    self.MASKED = MASKED
    self.d_model = d_model
    self.d_ff = d_ff
    self.voc_size = voc_size
    self.h = h

  def forward(self, metric):
    emb = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    # decoder block
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK = attn.forward(metric) + layernorm_1.forward(metric) + ffn.forward(metric) + layernorm_2.forward(metric)
    softmax = Softmax(M=self.MASKED, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return emb.forward(metric) + self.n_dec*BLOCK + softmax.forward(metric)

  def backward(self, metric):
    # decoder block
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK = attn.backward(metric) + layernorm_1.backward(metric) + ffn.backward(metric) + layernorm_2.backward(metric)
    softmax = Softmax(M=self.MASKED, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_dec*BLOCK + softmax.backward(metric)

  def weight_update(self, metric):
    # decoder block
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK = attn.weight_update(metric) + layernorm_1.weight_update(metric) + ffn.weight_update(metric) + layernorm_2.weight_update(metric)
    softmax = Softmax(M=self.MASKED, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_dec*BLOCK + softmax.weight_update(metric)

  def total_bp(self, metric):
    return self.forward(metric) + self.backward(metric) + self.weight_update(metric)

  def total_pep(self,metric):
    emb = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    return self.forward(metric) + emb.forward(metric) + self.forward(metric) + self.weight_update(metric)

  def total_mpep(self,metric):
    emb = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    return self.forward(metric) + emb.forward(metric) + 2*self.forward(metric) + self.weight_update(metric)
  
  def parameters(self):
    embedding = self.voc_size * self.d_model
    # decoder block
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK_PARAMS = attn.parameters() + layernorm_1.parameters() + ffn.parameters() + layernorm_2.parameters()
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    return embedding + self.n_dec * BLOCK_PARAMS 

  def activations_bp(self):
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK_MEM = attn.activations_bp() + layernorm_1.activations_bp() + ffn.activations_bp() + layernorm_2.activations_bp()
    softmax = Softmax(M=self.MASKED, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_dec*BLOCK_MEM + softmax.activations_bp()

  def activations_pep(self):
    attn = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    BLOCK_MEM = attn.activations_pep() + layernorm_1.activations_pep() + ffn.activations_pep() + layernorm_2.activations_pep()
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    return self.n_dec*BLOCK_MEM + softmax.activations_pep()
  
  def activations_mpep(self):
    embedding = self.M * self.voc_size + 2 * self.M * self.d_model
    attn_initial_a = self.M * self.d_model + 2 * self.M * self.d_model/self.h
    attn_initial_b = 2*self.M*self.d_model + self.M * self.d_model/self.h 
    attn_middle = 2*self.M*self.d_model + 2* self.M * self.d_model/self.h
    layernorm = 3 * self.M * self.d_model
    FFN = self.M * self.d_model + 2*self.M*self.d_ff
    last_softmax = self.M * self.d_model + self.M * self.voc_size
    act = [embedding, attn_initial_a, attn_initial_b, attn_middle, layernorm, FFN, last_softmax]
    return max(act)

  
  def print_stats(self):
    table = [['Train Complexity', 'BP', 'PEP', 'MEMPEP']]
    table.append(['MACs', '{:,}'.format(self.total_bp('macs')), '{:,}'.format(self.total_pep('macs')), '{:,}'.format(self.total_mpep('macs'))])
    table.append(['FLOPs', '{:,}'.format(self.total_bp('flops')), '{:,}'.format(self.total_pep('flops')), '{:,}'.format(self.total_mpep('flops'))])
    print(tabulate(table , headers='firstrow',tablefmt='fancy_grid'))

    table = [['Memory', 'BP', 'PEP', 'MEMPEP']]
    table.append(['Activations', '{:,}'.format(self.activations_bp()), '{:,}'.format(self.activations_pep()), '{:,}'.format(self.activations_mpep())])
    table.append(['Parameters', '{:,}'.format(self.parameters()), '{:,}'.format(self.parameters()), '{:,}'.format(self.parameters())])
    print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))
  
