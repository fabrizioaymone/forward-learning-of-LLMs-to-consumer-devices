from blocks.Attn import *
from blocks.FFN import *
from blocks.Layernorm import *
from blocks.Emb import *
from blocks.Softmax import *

from tabulate import tabulate



class encoder_decoder_model():
  def __init__(self,n_enc, N, n_dec, M, d_model, d_ff, h, voc_size):
    self.n_enc = n_enc
    self.N = N
    self.n_dec = n_dec
    self.M = M
    self.d_model = d_model
    self.d_ff = d_ff
    self.voc_size = voc_size
    self.h = h

  def forward(self, metric):
    emb_enc = Emb(M=self.N, d_model=self.d_model, voc_size=self.voc_size)
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.forward(metric) + layernorm_1.forward(metric) + ffn.forward(metric) + layernorm_2.forward(metric)

    emb_dec = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.forward(metric) + layernorm_1.forward(metric) + attn.forward(metric) + layernorm_2.forward(metric) + ffn.forward(metric) + layernorm_3.forward(metric)
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return emb_enc.forward(metric) + self.n_enc*ENC_BLOCK + emb_dec.forward(metric) + self.n_dec*DEC_BLOCK + softmax.forward(metric)

  def backward(self, metric):
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.backward(metric) + layernorm_1.backward(metric) + ffn.backward(metric) + layernorm_2.backward(metric)

    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.backward(metric) + layernorm_1.backward(metric) + attn.backward(metric) + layernorm_2.backward(metric) + ffn.backward(metric) + layernorm_3.backward(metric)
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_enc*ENC_BLOCK + self.n_dec*DEC_BLOCK + softmax.backward(metric)

  def weight_update(self, metric):
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.weight_update(metric) + layernorm_1.weight_update(metric) + ffn.weight_update(metric) + layernorm_2.weight_update(metric)

    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.weight_update(metric) + layernorm_1.weight_update(metric) + attn.weight_update(metric) + layernorm_2.weight_update(metric) + ffn.weight_update(metric) + layernorm_3.weight_update(metric)
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_enc*ENC_BLOCK + self.n_dec*DEC_BLOCK + softmax.weight_update(metric)

  def total_bp(self, metric):
    return self.forward(metric) + self.backward(metric) + self.weight_update(metric)

  def total_pep(self,metric):
    emb_enc = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    if metric == 'macs':
      proj_to_enc = 2*self.N*self.M*self.voc_size
    elif metric == 'flops':
      proj_to_enc = 4*self.N*self.M*self.voc_size + 6*self.M*self.N
    emb_dec = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    return self.forward(metric) + emb_dec.forward(metric) + proj_to_enc + emb_enc.forward(metric) + self.forward(metric) + self.weight_update(metric)

  def total_mpep(self,metric):
    emb_enc = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    if metric == 'macs':
      proj_to_enc = self.N*self.M*self.voc_size
    elif metric == 'flops':
      proj_to_enc = 2*self.N*self.M*self.voc_size + 6*self.M*self.N
    emb_dec = Emb(M=self.M, d_model=self.d_model, voc_size=self.voc_size)
    return self.forward(metric) + self.forward(metric) + (emb_dec.forward(metric) + proj_to_enc + emb_enc.forward(metric) + self.forward(metric)) + self.weight_update(metric)
   
  def parameters(self):
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.parameters() + layernorm_1.parameters() + ffn.parameters() + layernorm_2.parameters()

    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.parameters() + layernorm_1.parameters() + attn.parameters() + layernorm_2.parameters() + ffn.parameters() + layernorm_3.parameters()
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_enc*ENC_BLOCK + self.n_dec*DEC_BLOCK + softmax.parameters()

  def activations_bp(self):
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.activations_bp() + layernorm_1.activations_bp() + ffn.activations_bp() + layernorm_2.activations_bp()

    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.activations_bp() + layernorm_1.activations_bp() + attn.activations_bp() + layernorm_2.activations_bp() + ffn.activations_bp() + layernorm_3.activations_bp()
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_enc*ENC_BLOCK + self.n_dec*DEC_BLOCK + softmax.activations_bp()

  def activations_pep(self):
    # encoder block
    attn = Attn(M=self.N, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_2 = LayerNorm(M=self.N, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ENC_BLOCK = attn.activations_pep() + layernorm_1.activations_pep() + ffn.activations_pep() + layernorm_2.activations_pep()

    # decoder block
    attn_M = Attn(M=self.M, N=self.M, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_1 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    attn = Attn(M=self.M, N=self.N, d_model=self.d_model, voc_size=self.voc_size, h=self.h)
    layernorm_2 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    ffn = FFN(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    layernorm_3 = LayerNorm(M=self.M, d_model=self.d_model, d_ff=self.d_ff, voc_size=self.voc_size)
    DEC_BLOCK = attn_M.activations_pep() + layernorm_1.activations_pep() + attn.activations_pep() + layernorm_2.activations_pep() + ffn.activations_pep() + layernorm_3.activations_pep()
    softmax = Softmax(M=self.M, d_model=self.d_model, d_ff = self.d_ff, voc_size=self.voc_size)
    return self.n_enc*ENC_BLOCK + self.n_dec*DEC_BLOCK + softmax.activations_pep()
  
  def activations_mpep(self):
    # encoder
    projection_to_enc = self.M * self.voc_size + self.N* self.voc_size
    embedding_enc = self.M * self.voc_size + self.N * self.voc_size + 2 * self.N * self.d_model
    attn_initial_a_enc = self.M * self.voc_size + self.N * self.d_model + 2 * self.N * self.d_model/self.h
    attn_initial_b_enc = self.M * self.voc_size + 2*self.N*self.d_model + self.N * self.d_model/self.h 
    attn_middle_enc = self.M * self.voc_size + 2*self.N*self.d_model + 2* self.N * self.d_model/self.h
    layernorm_enc = self.M * self.voc_size + 3 * self.N * self.d_model
    FFN_enc = self.M * self.voc_size + self.N * self.d_model + 2*self.N*self.d_ff
    # decoder
    embedding_dec = self.M * self.voc_size + 2 * self.M * self.d_model
    attn_1_initial_a_dec = self.N*self.d_model + self.M * self.d_model + 2 * self.M * self.d_model/self.h
    attn_1_initial_b_dec = self.N*self.d_model + 2*self.M*self.d_model + self.M * self.d_model/self.h 
    attn_1_middle_dec = self.N*self.d_model + 2*self.M*self.d_model + 2* self.M * self.d_model/self.h
    attn_2_initial_q_a_dec = self.N*self.d_model + self.M * self.d_model + 2 * self.M * self.d_model/self.h
    attn_2_initial_kv_a_dec = self.N*self.d_model + self.N * self.d_model + 2 * self.N * self.d_model/self.h
    attn_2_initial_q_b_dec = self.N*self.d_model + 2*self.M*self.d_model + self.M * self.d_model/self.h 
    attn_2_initial_kv_b_dec = self.N*self.d_model + 2*self.N*self.d_model + self.M * self.d_model/self.h 
    attn_2_middle_dec = self.N*self.d_model + 2*self.M*self.d_model + 2* self.M * self.d_model/self.h
    layernorm_dec = 3 * self.M * self.d_model
    FFN_dec = self.M * self.d_model + 2*self.M*self.d_ff
    last_softmax = self.M * self.d_model + self.M * self.voc_size
    act = [projection_to_enc, embedding_enc, embedding_dec, attn_initial_a_enc, attn_initial_b_enc, attn_middle_enc, layernorm_enc, FFN_enc, attn_1_initial_a_dec, attn_1_initial_b_dec, attn_1_middle_dec, attn_2_initial_q_a_dec, attn_2_initial_kv_a_dec, attn_2_initial_q_b_dec, attn_2_initial_kv_b_dec, attn_2_middle_dec, layernorm_dec, FFN_dec,last_softmax]
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