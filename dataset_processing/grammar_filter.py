from dataclasses import dataclass, field
from typing import Protocol

from lark import Lark


class GrammerFilterI(Protocol):
    def parse(self, comment: str) -> bool: ...

    def __call__(self, comment: str) -> bool: ...


@dataclass(slots=True)
class GrammarFilter:
    grammar: str
    parser: Lark = field(init=False)

    def __post_init__(self):
        self.parser = Lark(self.grammar)

    def parse(self, comment: str) -> bool:
        try:
            self.parser.parse(text=comment)
        except Exception:
            return False
        else:
            return True

    def __call__(self, comment: str) -> bool:
        return self.parse(comment)
