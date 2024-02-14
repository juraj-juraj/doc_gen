import logging


class Model:
    def __init__(self, *args, **kwargs) -> None:
        self.docstring = "This is a docstring"

    def generate(self, code: str) -> str:
        logging.debug("Generating docstring for code")
        return self.docstring
