###############
# This needs more attention
# Not working at the moment
###############

import torch
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

# initialize the model

model_path = "Phind/Phind-CodeLlama-34B-v2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# HumanEval helper


def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(
        inputs.input_ids.to("cuda"),
        max_new_tokens=384,
        do_sample=True,
        top_p=0.75,
        top_k=40,
        temperature=0.1,
    )
    completion = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion


# perform HumanEval
problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)

# run `evaluate_functional_correctness samples.jsonl` in your HumanEval code sandbox
