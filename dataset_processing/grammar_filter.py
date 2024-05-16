from dataclasses import dataclass, field
from typing import Protocol
from lark import Lark


class GrammerFilterI(Protocol):
    def parse(self, comment: str) -> bool:
        ...

    def __call__(self, comment: str) -> bool:
        ...


@dataclass(slots=True)
class GrammarFilter:
    grammar: str
    parser: Lark = field(init=False)

    def __post_init__(self):
        """
        Initialize the parser with the specified grammar.
        """
        self.parser = Lark(self.grammar)

    def parse(self, comment: str) -> bool:
        """
        Parse the given comment using the specified parser.

        Args:
            comment (str): The comment to be parsed.

        Returns:
            bool: True if the comment was successfully parsed, False otherwise.
        """
        try:
            self.parser.parse(text=comment)
        except Exception:
            return False
        else:
            return True

    def __call__(self, comment: str) -> bool:
        """
        Parse the given comment and return a boolean value.

        Args:
            comment (str): The comment to be parsed.

        Returns:
            bool: True if the comment is successfully parsed, False otherwise.
        """
        return self.parse(comment)
