#%%
import torch as th
from model2 import BiSyn_GAT_plus
from snippet import *
import re
import os

def extract_first_five(text):
    match = re.match(r'.{1,20}', text)
    return match.group() if match else ''
args = get_parameter()
model = BiSyn_GAT_plus(args).to(device=args.device)
n=0
for name,param in model.named_parameters():
    if extract_first_five(name)=="intra_context_module":
        n=n+1
    result = extract_first_five(name)
    if result=="intra_context_module":
        param.requires_grad=False

checkpoint_PATH = './data/V2/MAMS/checkpoint/checkpoint4.1.pth'
checkpoint=th.load(checkpoint_PATH)

model.load_state_dict(checkpoint['model.state_dict()'],strict=False)
th.save(
    {
        'model.state_dict()': model.state_dict(),
        'classifier': model.classifier.state_dict(),
    },
    os.path.join(
        './data/V2/MAMS/checkpoint', 'checkpoint4.pth'
    )
)

checkpoint_PATH = './data/V2/MAMS/checkpoint/checkpoint4.pth'
checkpoint2=th.load(checkpoint_PATH)

#print(checkpoint.load_state_dict()['intra_context_module']['context_encoder.embeddings.word_embeddings.weight'])
#model.intra_context_module.load_state_dict(checkpoint['intra_context_module'],strict=False)
#冻结参数
for name,param in model.named_parameters():
    result = extract_first_five(name)
    if result=="intra_context_module":
        param.requires_grad=False

# 不传参数
checkpoint['']
# for name,param in model.named_parameters():
#     print(name)
#     result = extract_first_five(name)
#     if result=="intra_context_module":
#         param.requires_grad=False
#
#
