# %%

import torch
from model import Transformer
from tokenizer import SpecialTokens, decode_batch, get_encoding

# %%

model = Transformer().eval().cuda()
model.load_state_dict(torch.load("./ckpts/model_01000000.pt"))

# %%

encoding = get_encoding()

# %%

prompt = f"{SpecialTokens.BEGIN}When Alex woke up from sleep, he found himself trapped in a room."
inputs = encoding.encode(prompt, allowed_special="all")
inputs = torch.tensor([inputs]).cuda().repeat(5, 1)


# %%

outputs = model.sample(inputs, max_length=128, top_p=0.9, use_cache=True)
for seq in decode_batch(encoding, torch.cat([inputs, outputs], dim=-1), trim=True):
    print(f"• {seq}\n")

# %%
