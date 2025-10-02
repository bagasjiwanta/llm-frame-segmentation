import os

import torch
from transformers import AutoModelForVision2Seq

model = AutoModelForVision2Seq.from_pretrained(
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5", trust_remote_code=True
).vlm
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/xgenmm.pt")
