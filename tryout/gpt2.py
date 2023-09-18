import logging

from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, device: str):
        """
        General purpose model for generating text.

        Args:
            device (str): device to run inference on
        """
        self._device = device

        checkpoint = "gpt2"
        logging.debug(f"Loading checkpoint {checkpoint}...")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logging.info("Tokenizer loaded.")
        self._model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self._device)
        logging.info("Model loaded.")

    def generate(self, prompt: str, max_length: int = 150) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        output_sequences = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
        return self._tokenizer.decode(output_sequences[0], skip_special_tokens=True)
