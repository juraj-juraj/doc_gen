###################
# This doesnt work on computer DELL G5 5590
###################

# import logging

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = "tiiuae/falcon-7b-instruct"
# logging.basicConfig(level=logging.INFO)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# logging.info(f"Using device {device}")

# tokenizer = AutoTokenizer.from_pretrained(model)
# model = AutoModelForCausalLM.from_pretrained(
#     model,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# ).to(device)
# logging.info("Model loaded.")

# inputs = tokenizer("Generate poem for nice weather", return_tensors="pt").to(device)
# generated_text = model.generate(**inputs, max_new_tokens=20, do_sample=True)
# print(generated_text)

###################
# This works on computer DELL G5 5590
###################

import logging

import torch
from transformers import AutoTokenizer, FalconForCausalLM

logging.basicConfig(level=logging.DEBUG)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.debug(f"Using device {device}")

tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
logging.debug("Tokenizer loaded")
model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b").to(device)
logging.debug("Model loaded")

inputs = tokenizer("Generate poem for nice weather", return_tensors="pt").to(device)
generated_text = model.generate(**inputs, max_new_tokens=100, do_sample=True)

text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(text)
