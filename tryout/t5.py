#######################
# This works on Dell G5 5590
# Usable
#######################

import logging

from transformers import T5ForConditionalGeneration, T5Tokenizer


class Model:
    def __init__(self, device: str) -> None:
        """
        Model is best for translations for now.
        Example sentence: "translate English to German: Hello, how are you?"

        Args:
            device (str): Device to run inference on
        """
        self._device = device

        checkpoint = "t5-small"
        logging.debug(f"Loading checkpoint {checkpoint}...")
        self._tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        logging.info("Tokenizer loaded.")
        self._model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(self._device)
        logging.info("Model loaded.")

    def generate(self, prompt: str, max_length: int = 150) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        output_sequences = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
        return self._tokenizer.decode(output_sequences[0], skip_special_tokens=True)
