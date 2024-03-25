import ast
import logging
import pathlib

import pandas as pd


def remove_docstring(node: ast.AST) -> ast.AST:
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        node.body.pop(0)
    return node


class ExtractDocstrings(ast.NodeTransformer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._corpus = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        docs = ast.get_docstring(node, clean=True).split("\n\n\n")
        ast_fce = remove_docstring(node)
        fce = ast.unparse(ast_fce)
        self._corpus.append({"functions": fce, "docstrings": docs})
        return self.generic_visit(node)

    @property
    def corpus(self):
        return pd.DataFrame(self._corpus, columns=["functions", "docstrings"])


def load_corpus(corpus_path: pathlib.Path) -> pd.DataFrame:
    raw_code = corpus_path.read_text(encoding="utf-8")
    logging.debug(f"Loaded corpus data of length: {len(raw_code)}")
    tree = ast.parse(raw_code)
    extract_docstring = ExtractDocstrings()
    extract_docstring.visit(tree)
    corpus = extract_docstring.corpus
    logging.info(f"Loaded {len(corpus)} samples from corpus")
    return corpus


def save_annotations(data: pd.DataFrame, output: pathlib.Path):
    annotated = [f"\"\"\"{row['predictions']}\"\"\"\n{row['functions']}" for _, row in data.iterrows()]
    txt_data = "\n\n\n".join(annotated)
    output.write_text(txt_data, encoding="utf-8")
