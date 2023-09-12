from encoder_only_model import encoder_only_model
from decoder_only_model import decoder_only_model
from encoder_decoder_model import encoder_decoder_model

from tabulate import tabulate
import numpy as np
from matplotlib import pyplot as plt
import scienceplots

plt.style.use(['science','ieee', 'no-latex'])


fig = plt.figure(constrained_layout=True, figsize=(10,8.5))
subfigs = fig.subfigures(nrows=3, ncols=1)

font_var = 10
ctx = 2048
x = np.arange(1, ctx+1)


#DistilBERT
MACCs_BP = []
MACCs_PEP = []
MACCs_MPE = []
FLOPs_BP = []
FLOPs_PEP = []
FLOPs_MPE = []
ACT_BP = []
ACT_PEP = []
ACT_MPE = []

for i in np.arange(1, ctx+1):
    config_distilBERT = {'n_dec' : 6, 'M' : i, 'MASKED' : i, 'd_model' : 768, 'd_ff' : 768*4, 'h' : 12, 'voc_size' : 30522}
    BERT = encoder_only_model(**config_distilBERT)
    MACCs_BP.append(BERT.total_bp('macs')/(10**9))
    MACCs_PEP.append(BERT.total_pep('macs')/(10**9))
    MACCs_MPE.append(BERT.total_mpep('macs')/(10**9))
    FLOPs_BP.append(BERT.total_bp('flops')/(10**9))
    FLOPs_PEP.append(BERT.total_pep('flops')/(10**9))
    FLOPs_MPE.append(BERT.total_mpep('flops')/(10**9))
    ACT_BP.append(BERT.activations_bp()/(10**6))
    ACT_PEP.append(BERT.activations_pep()/(10**6))
    ACT_MPE.append(BERT.activations_mpep()/(10**6))
print("DistilBERT")


BERT_ACT_BP = []
BERT_ACT_PEP = []
BERT_ACT_MPE = []


BERT_MACCs_BP = []
BERT_MACCs_PEP = []
BERT_MACCs_MPE = []

for i in [32, 128, 512, 1024, 2048]:
    BERT_ACT_BP.append(ACT_BP[i-1])
    BERT_ACT_PEP.append(ACT_PEP[i-1])
    BERT_ACT_MPE.append(ACT_MPE[i-1])

for i in [32, 128, 512, 1024, 2048]:
    BERT_MACCs_BP.append(MACCs_BP[i-1])
    BERT_MACCs_PEP.append(MACCs_PEP[i-1])
    BERT_MACCs_MPE.append(MACCs_MPE[i-1])
    
print("MACCs")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((MACCs_PEP[i-1]/MACCs_BP[i-1]-1)*100), '{:,}'.format((MACCs_MPE[i-1]/MACCs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("FLOPS")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((FLOPs_PEP[i-1]/FLOPs_BP[i-1]-1)*100), '{:,}'.format((FLOPs_MPE[i-1]/FLOPs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("ACT")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((ACT_PEP[i-1]/ACT_BP[i-1]-1)*100), '{:,}'.format((ACT_MPE[i-1]/ACT_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))



flag = False
flag_ = False
for i in range(ctx):
    if MACCs_BP[i]>MACCs_MPE[i] and not flag:
        print(f"Distilbet MACCS BE is {x[i]}")
        flag = True
for i in range(ctx):
    if FLOPs_BP[i]>FLOPs_MPE[i] and not flag_:
        print(f"Distilbet FLOPS BE is {x[i]}")
        flag_ = True

subfigs[0].suptitle('DistilBERT')
axs = subfigs[0].subplots(nrows=1, ncols=3)



axs[0].plot(x, MACCs_BP, label = 'BP')
axs[0].plot(x, MACCs_PEP, label = 'PEP')
axs[0].plot(x, MACCs_MPE, label = 'MPEP')
axs[0].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[0].set(ylabel='MACCs (B)')
axs[0].set_title('a)', loc='left')


handles, labels = axs[0].get_legend_handles_labels()

fig.legend(handles, labels, loc="outside lower center", ncol=3)

axs[1].plot(x, FLOPs_BP, label = 'BP')
axs[1].plot(x, FLOPs_PEP, label = 'PEP')
axs[1].plot(x, FLOPs_MPE, label = 'MPEP')
axs[1].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[1].set(ylabel='FLOPs (B)')
axs[1].set_title('b)', loc='left')



axs[2].plot(x, ACT_BP, label = 'BP')
axs[2].plot(x, ACT_PEP, label = 'PEP')
axs[2].plot(x, ACT_MPE, label = 'MPEP')
axs[2].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[2].set(ylabel='ACTIVATIONS (MB)')
axs[2].set_title('c)', loc='left')


#GPT-3 Small
MACCs_BP = []
MACCs_PEP = []
MACCs_MPE = []
FLOPs_BP = []
FLOPs_PEP = []
FLOPs_MPE = []
ACT_BP = []
ACT_PEP = []
ACT_MPE = []

for i in np.arange(1, ctx+1):
    #the voc size is assumed to be the same of GPT-2
    config_gpt3 = {'n_dec' : 12, 'M' : i, 'd_model' : 784, 'd_ff' : 768*4, 'h' : 12, 'voc_size' : 50257}
    GPT3 = decoder_only_model(**config_gpt3)
    MACCs_BP.append(GPT3.total_bp('macs')/(10**9))
    MACCs_PEP.append(GPT3.total_pep('macs')/(10**9))
    MACCs_MPE.append(GPT3.total_mpep('macs')/(10**9))
    FLOPs_BP.append(GPT3.total_bp('flops')/(10**9))
    FLOPs_PEP.append(GPT3.total_pep('flops')/(10**9))
    FLOPs_MPE.append(GPT3.total_mpep('flops')/(10**9))
    ACT_BP.append(GPT3.activations_bp()/(10**6))
    ACT_PEP.append(GPT3.activations_pep()/(10**6))
    ACT_MPE.append(GPT3.activations_mpep()/(10**6))


GPT_ACT_BP = []
GPT_ACT_PEP = []
GPT_ACT_MPE = []

GPT_MACCs_BP = []
GPT_MACCs_PEP = []
GPT_MACCs_MPE = []


for i in [32, 128, 512, 1024, 2048]:
    GPT_ACT_BP.append(ACT_BP[i-1])
    GPT_ACT_PEP.append(ACT_PEP[i-1])
    GPT_ACT_MPE.append(ACT_MPE[i-1])

for i in [32, 128, 512, 1024, 2048]:
    GPT_MACCs_BP.append(MACCs_BP[i-1])
    GPT_MACCs_PEP.append(MACCs_PEP[i-1])
    GPT_MACCs_MPE.append(MACCs_MPE[i-1])
    
print("GPT3")
print("MACCs")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((MACCs_PEP[i-1]/MACCs_BP[i-1]-1)*100), '{:,}'.format((MACCs_MPE[i-1]/MACCs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("FLOPS")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((FLOPs_PEP[i-1]/FLOPs_BP[i-1]-1)*100), '{:,}'.format((FLOPs_MPE[i-1]/FLOPs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("ACT")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((ACT_PEP[i-1]/ACT_BP[i-1]-1)*100), '{:,}'.format((ACT_MPE[i-1]/ACT_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))


flag = False
flag_ = False
for i in range(ctx):
    if MACCs_BP[i]>MACCs_MPE[i] and not flag:
        print(f"GPT3 MACCS BE is {x[i]}")
        flag = True
for i in range(ctx):
    if FLOPs_BP[i]>FLOPs_MPE[i] and not flag_:
        print(f"GPT3 FLOPS BE is {x[i]}")
        flag_ = True

subfigs[1].suptitle('GPT-3 Small')
axs = subfigs[1].subplots(nrows=1, ncols=3)

axs[0].plot(x, MACCs_BP, label = 'BP')
axs[0].plot(x, MACCs_PEP, label = 'PEP')
axs[0].plot(x, MACCs_MPE, label = 'MPEP')
axs[0].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[0].set(ylabel='MACCs (B)')
axs[0].set_title('d)', loc='left')

axs[1].plot(x, FLOPs_BP, label = 'BP')
axs[1].plot(x, FLOPs_PEP, label = 'PEP')
axs[1].plot(x, FLOPs_MPE, label = 'MPEP')
axs[1].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[1].set(ylabel='FLOPs (B)')
axs[1].set_title('e)', loc='left')


axs[2].plot(x, ACT_BP, label = 'BP')
axs[2].plot(x, ACT_PEP, label = 'PEP')
axs[2].plot(x, ACT_MPE, label = 'MPEP')
axs[2].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[2].set(ylabel='ACTIVATIONS (MB)')
axs[2].set_title('f)', loc='left')



#AlexaTM
MACCs_BP = []
MACCs_PEP = []
MACCs_MPE = []
FLOPs_BP = []
FLOPs_PEP = []
FLOPs_MPE = []
ACT_BP = []
ACT_PEP = []
ACT_MPE = []

for i in np.arange(1, ctx+1):
    config_alexatm = {'n_enc' : 46, 'N' : 100, 'n_dec' : 32, 'M' : i, 'd_model' : 4096, 'd_ff' : 4096*4, 'h' : 32, 'voc_size': 150000}
    AlexaTM = encoder_decoder_model(**config_alexatm)
    MACCs_BP.append(AlexaTM.total_bp('macs')/(10**12))
    MACCs_PEP.append(AlexaTM.total_pep('macs')/(10**12))
    MACCs_MPE.append(AlexaTM.total_mpep('macs')/(10**12))
    FLOPs_BP.append(AlexaTM.total_bp('flops')/(10**12))
    FLOPs_PEP.append(AlexaTM.total_pep('flops')/(10**12))
    FLOPs_MPE.append(AlexaTM.total_mpep('flops')/(10**12))
    ACT_BP.append(AlexaTM.activations_bp()/(10**6))
    ACT_PEP.append(AlexaTM.activations_pep()/(10**6))
    ACT_MPE.append(AlexaTM.activations_mpep()/(10**6))

ALEXA_ACT_BP = []
ALEXA_ACT_PEP = []
ALEXA_ACT_MPE = []

ALEXA_MACCs_BP = []
ALEXA_MACCs_PEP = []
ALEXA_MACCs_MPE = []


for i in [32, 128, 512, 1024, 2048]:
    ALEXA_ACT_BP.append(ACT_BP[i-1])
    ALEXA_ACT_PEP.append(ACT_PEP[i-1])
    ALEXA_ACT_MPE.append(ACT_MPE[i-1])

for i in [32, 128, 512, 1024, 2048]:
    ALEXA_MACCs_BP.append(MACCs_BP[i-1])
    ALEXA_MACCs_PEP.append(MACCs_PEP[i-1])
    ALEXA_MACCs_MPE.append(MACCs_MPE[i-1])

print("AlexaTM")
print("MACCs")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((MACCs_PEP[i-1]/MACCs_BP[i-1]-1)*100), '{:,}'.format((MACCs_MPE[i-1]/MACCs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("FLOPS")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((FLOPs_PEP[i-1]/FLOPs_BP[i-1]-1)*100), '{:,}'.format((FLOPs_MPE[i-1]/FLOPs_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))

print("ACT")
table = [['n_ctx', 'PEP', 'MEMPEP']]
for i in [32, 128, 512, 2048]:
    table.append([f'{i}', '{:,}'.format((ACT_PEP[i-1]/ACT_BP[i-1]-1)*100), '{:,}'.format((ACT_MPE[i-1]/ACT_BP[i-1]-1)*100)])

print(tabulate(table,headers='firstrow',tablefmt='fancy_grid', numalign="right"))


flag = False
flag_ = False
for i in range(ctx):
    if MACCs_BP[i]>MACCs_MPE[i] and not flag:
        print(f"AlexaTM MACCS BE is {x[i]}")
        flag = True
    if FLOPs_BP[i]>FLOPs_MPE[i] and not flag_:
        print(f"AlexaTM FLOPS BE is {x[i]}")
        flag_ = True


subfigs[2].suptitle('AlexaTM')
axs = subfigs[2].subplots(nrows=1, ncols=3)

axs[0].plot(x, MACCs_BP, label = 'BP')
axs[0].plot(x, MACCs_PEP, label = 'PEP')
axs[0].plot(x, MACCs_MPE, label = 'MPEP')
axs[0].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[0].set(ylabel='MACCs (T)')
axs[0].set_title('g)', loc='left')


axs[1].plot(x, FLOPs_BP, label = 'BP')
axs[1].plot(x, FLOPs_PEP, label = 'PEP')
axs[1].plot(x, FLOPs_MPE, label = 'MPEP')
axs[1].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[1].set(ylabel='FLOPs (T)')
axs[1].set_title('h)', loc='left')


axs[2].plot(x, ACT_BP, label = 'BP')
axs[2].plot(x, ACT_PEP, label = 'PEP')
axs[2].plot(x, ACT_MPE, label = 'MPEP')
axs[2].set_xlabel(xlabel= r'$n_{ctx}$', fontsize=font_var)
axs[2].set(ylabel='ACTIVATIONS (MB)')
axs[2].set_title('i)', loc='left')

plt.show()

# activations graph
plt.style.use(['science','ieee', 'no-latex', 'vibrant'])

n_ctx_ord = 3
BERT_PARAMS= 66
models = ("BERT")
learning = ("BP", "PEP", "MPE")
mem = {"params": np.array([66, 66, 66]), "act": np.array([BERT_ACT_BP[n_ctx_ord], BERT_ACT_PEP[n_ctx_ord], BERT_ACT_MPE[n_ctx_ord]])}


fig, axs  = plt.subplots(1,3, figsize=(15,4))
bottom = np.zeros(3)

for label, values in mem.items():
    p = axs[0].bar(learning, values, 0.5, label = label, bottom=bottom )
    bottom += values


axs[0].axhline(y=128, color='r', linestyle='--')
axs[0].text(0.6, 130, "MTIA v1 on-chip SRAM", color='grey', fontsize=6, transform=axs[0].get_yaxis_transform())

axs[0].axhline(y=256, color='r', linestyle='--')
axs[0].text(0.6, 258, "Raspberry P1 DRAM", color='grey', fontsize=6, transform=axs[0].get_yaxis_transform())

axs[0].text(0.6, 3, "STM32H733ZGT6", color='grey', fontsize=5, transform=axs[0].get_yaxis_transform())
axs[0].axhline(y=1, color='r', linestyle='--')

axs[0].set_ylabel("Memory footprint (MB)")
axs[0].set_title("DistilBERT")
axs[0].legend(loc="upper right")
axs[0].set_ylim(top = 300)



GPT_PARAMS= 128
models = ("GPT")
learning = ("BP", "PEP", "MPE")
mem = {"params": np.array([66, 66, 66]), "act": np.array([GPT_ACT_BP[n_ctx_ord], GPT_ACT_PEP[n_ctx_ord], GPT_ACT_MPE[n_ctx_ord]])}


bottom = np.zeros(3)

for label, values in mem.items():
    p = axs[1].bar(learning, values, 0.5, label = label, bottom=bottom )
    bottom += values



axs[1].axhline(y=128, color='r', linestyle='--')
axs[1].text(0.6, 130, "MTIA v1 on-chip SRAM", color='grey', fontsize=6, transform=axs[1].get_yaxis_transform())

axs[1].axhline(y=256, color='r', linestyle='--')
axs[1].text(0.6, 258, "Raspberry P1 DRAM", color='grey', fontsize=6, transform=axs[0].get_yaxis_transform())

axs[1].text(0.6, 3, "STM32H733ZGT6", color='grey', fontsize=5, transform=axs[1].get_yaxis_transform())
axs[1].axhline(y=1, color='r', linestyle='--')

axs[1].set_title("GPT-3 Small")
axs[1].legend(loc="upper right")
axs[1].set_ylim(top = 300)


AlexaTM_PARAMS= 19750
models = ("AlexaTM")
learning = ("BP", "PEP", "MPE")
mem = {"params": np.array([19750, 19750, 19750]), "act": np.array([ALEXA_ACT_BP[n_ctx_ord], ALEXA_ACT_PEP[n_ctx_ord], ALEXA_ACT_MPE[n_ctx_ord]])}


bottom = np.zeros(3)

for label, values in mem.items():
    p = axs[2].bar(learning, values, 0.5, label = label, bottom=bottom )
    bottom += values

axs[2].axhline(y=24000, color='r', linestyle='--')
axs[2].text(0.6, 24200, "Snapdragon 8 Gen 2 SDRAM", color='grey', fontsize=7, transform=axs[2].get_yaxis_transform())



axs[2].set_title("AlexaTM")
axs[2].legend(loc="upper right")
axs[2].set_ylim(top = 35000)

plt.show()

# latency graph

fig, axs  = plt.subplots(1,3, figsize=(10,3))

axs[0].bar(learning, (BERT_MACCs_BP[n_ctx_ord]*(10**9)/(700*10**6)/60, BERT_MACCs_PEP[n_ctx_ord]*(10**9)/(700*10**6)/60, BERT_MACCs_MPE[n_ctx_ord]*(10**9)/(700*10**6)/60), 0.5)
axs[0].set_title("DistilBERT")
axs[0].set_ylim(ymax=20)
axs[0].set_ylabel('Latency (min)')

axs[1].bar(learning, (GPT_MACCs_BP[n_ctx_ord]*(10**9)/(700*10**6)/60, GPT_MACCs_PEP[n_ctx_ord]*(10**9)/(700*10**6)/60, GPT_MACCs_MPE[n_ctx_ord]*(10**9)/(700*10**6)/60), 0.5)
axs[1].set_title("GPT-3 Small")
axs[1].set_ylim(ymax=20)

axs[2].bar(learning, (ALEXA_MACCs_BP[n_ctx_ord]*(10**12)/(2*8*10**9)/60, ALEXA_MACCs_PEP[n_ctx_ord]*(10**12)/(2*8*10**9)/60, ALEXA_MACCs_MPE[n_ctx_ord]*(10**12)/(2*8*10**9)/60), 0.5)
axs[2].set_title("AlexaTM")
axs[2].set_ylim(ymax=50)

plt.show()
