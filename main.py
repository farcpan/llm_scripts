import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[START]load model")
start = time.time()
model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
print(f"[END]load model: {time.time() - start} [sec]")

print("[START]tokenizer")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
print(f"[END]tokenizer: {time.time() - start} [sec]")

print("[START]inference")
inputs = tokenizer("AIによって私達の暮らしは、", return_tensors="pt").to(model.device)
with torch.no_grad():
    start = time.time()
    tokens = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )
    
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(f"[END]inference: {time.time() - start} [sec]")
print(output)
