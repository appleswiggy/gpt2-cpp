"""
This file converts HuggingFace GPT-2 weights to a flat binary file.

Use `uv` to add transformers, torch and numpy and run this file.
Note: Pass in additional command line parameter to specify which model to download and export (default is gpt2 (small))

Binary format:
    [Header] 7 int32 values: magic, version, n_layer, n_head, n_embd, vocab_size, block_size (context size)
    [Weights] all float32 tensors concatenated in a fixed order (see below)
"""

import sys
import struct
import numpy as np

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
MAGIC = 0x67707432 # "gpt2" in hex - so C++ code can validate the file
VERSION = 1

# load the model
print(f"Loading {MODEL_NAME} from HuggingFace...")
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
config = model.config
sd = model.state_dict()

print(f"n_layer={config.n_layer}, n_head={config.n_head}, "
      f"n_embd={config.n_embd}, vocab_size={config.vocab_size}, "
      f"block_size={config.n_positions}")

# define weight order (see docs to see the shape and size of each matrix and where they are used)

weight_names = []
weight_names.append("transformer.wte.weight") # Token embeddings weight matrix [vocab_size, n_embd]
weight_names.append("transformer.wpe.weight") # Positional encodings matrix [block_size, n_embd]

for i in range(config.n_layer):
    p = f"transformer.h.{i}"
    weight_names += [
        f"{p}.ln_1.weight", f"{p}.ln_1.bias", # LayerNorm 1 
        f"{p}.attn.c_attn.weight", f"{p}.attn.c_attn.bias", # QKV projection
        f"{p}.attn.c_proj.weight", f"{p}.attn.c_proj.bias", # Attention output
        f"{p}.ln_2.weight", f"{p}.ln_2.bias", # LayerNorm 2
        f"{p}.mlp.c_fc.weight", f"{p}.mlp.c_fc.bias", # MLP up-project
        f"{p}.mlp.c_proj.weight", f"{p}.c_proj.bias", # MLP down-project
    ]

weight_names += ["transformer.ln_f.weight", "transformer.ln_f.bias"]
# Note: lm_head.weight == wte.weight (weight tying), so not stored separately.

# write the binary file
out_path = f"models/{MODEL_NAME.replace('/', '_')}.bin"
print(f"Writing {out_path}...")

total_params = 0
with open(out_path, "wb") as f:
    f.write(struct.pack("iiiiiii",
                        MAGIC, VERSION, config.n_layer, config.n_head,
                        config.n_embd, config.vocab_size, config.n_positions))
    
    for name in weight_names:
        w = sd[name].float().numpy().flatten()
        total_params += w.size
        f.write(w.tobytes())

print(f"Done! {total_params:,} parameters ({total_params * 4 / 1e6:.1f} MB)")
print(f"Saved to: {out_path}")

# also export tokenizer (we read the json directly in c++ so just save_pretrained)
import os
from transformers import AutoTokenizer

print("Exporting tokenizer...")
tok_dir = "models/tokenizer"
os.makedirs(tok_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(tok_dir)
print(f"Tokenizer saved to: {tok_dir}/")