from typing import Optional, Protocol


class Model(Protocol):
    def __init__(self, device: str) -> None:
        ...

    def generate(self, prompt: str, max_length: Optional[int]) -> str:
        ...
