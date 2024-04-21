# Load model directly
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "EleutherAI/pile-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device="cuda")

prompt = """Generate docstring to python function:
def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict:
    evaluator = evaluate.load("meteor", num_process=self.n_workers)
    result = evaluator.compute(predictions=preds, references=refs)
    result["score"] = result["meteor"]
    return result
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device="cuda")
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
print(f"Input: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}")

print(f"Output: {tokenizer.decode(output_sequences[0], skip_special_tokens=True)}")
