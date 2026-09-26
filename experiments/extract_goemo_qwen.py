# -*- coding: utf-8 -*-
"""GoEmotions -> Qwen3-0.6b 最后层均池化嵌入(GPU2): LLM表征下的先验偏移实验。
输出 /root/autodl-tmp/goemo_qwen.npz"""
import os
import numpy as np, torch, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
t0=time.time()
ds=load_dataset("google-research-datasets/go_emotions","simplified")
NC=28
def to_xy(split,code):
    txt=[r["text"] for r in ds[split]]
    Y=np.zeros((len(txt),NC),np.int8)
    for i,r in enumerate(ds[split]):
        for l in r["labels"]: Y[i,l]=1
    return txt,Y,np.full(len(txt),code,np.int8)
parts=[to_xy("train",0),to_xy("validation",1),to_xy("test",2)]
texts=sum([p[0] for p in parts],[]); Y=np.concatenate([p[1] for p in parts]); SP=np.concatenate([p[2] for p in parts])
dev="cuda:2" if torch.cuda.device_count()>2 else "cuda:0"
MP=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-0.6B")
tok=AutoTokenizer.from_pretrained(MP)
model=AutoModel.from_pretrained(MP,torch_dtype=torch.float16).to(dev).eval()
H=model.config.hidden_size
feats=np.zeros((len(texts),H),np.float16)
B=128
with torch.no_grad():
    for i in range(0,len(texts),B):
        b=tok(texts[i:i+B],padding=True,truncation=True,max_length=64,return_tensors="pt").to(dev)
        h=model(**b).last_hidden_state; m=b["attention_mask"].unsqueeze(-1)
        e=(h*m).sum(1)/m.sum(1)
        feats[i:i+B]=torch.nn.functional.normalize(e.float(),dim=-1).cpu().numpy().astype(np.float16)
        if i%(B*40)==0: print(f"  {i}/{len(texts)} {time.time()-t0:.0f}s",flush=True)
out_path=os.environ.get("GOEMO_QWEN_OUTPUT", "/root/autodl-tmp/goemo_qwen.npz")
np.savez_compressed(out_path,feats=feats,labels=Y,split=SP)
print(f"QWEN_EMB_DONE t={time.time()-t0:.0f}s",flush=True)
