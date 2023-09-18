import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Model:
    def __init__(self, device: str) -> None:
        """Initialize model and tokenizer
        This works on computer DELL G5 5590
        Model summary:
                codet5p-770m and codet5p-220m works well, generating code
                codet5-small can be run, but not generating anything usable
                codet5p-2b doesnt run, due to errors I cannot solve, seems it is too performance demanding
        """
        self._device = device

        checkpoint = "Salesforce/codet5p-770m"
        logging.debug(f"Loading checkpoint {checkpoint}...")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logging.info("Tokenizer loaded.")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self._device)
        logging.info("Model loaded.")

    def generate(self, prompt: str, max_length: int = 150) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        output_sequences = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
        return self._tokenizer.decode(output_sequences[0], skip_special_tokens=True)
