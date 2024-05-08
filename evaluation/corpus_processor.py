import ast
import logging
import pathlib
import pandas as pd


def remove_docstring(node: ast.AST) -> ast.AST:
    """
    Removes the docstring from the given AST node if it is an expression and a Constant value.

    Parameters:
        node (ast.AST): The AST node to be processed.

    Returns:
        ast.AST: The modified AST node with the docstring removed if it is an expression and a Constant value.
    """
    if isinstance(node.body[0], ast.Expr) and isinstance(
        node.body[0].value, ast.Constant
    ):
        node.body.pop(0)
    return node


class ExtractDocstrings(ast.NodeTransformer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._corpus = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """
        Visit a FunctionDef node and extract documentation from it.

        Args:
            self: The current instance of the class.
            node (ast.FunctionDef): The FunctionDef node to visit.

        Returns:
            ast.AST: The AST after extracting documentation from the FunctionDef node.
        """
        docs = ast.get_docstring(node, clean=True).split("\n\n\n")
        ast_fce = remove_docstring(node)
        fce = ast.unparse(ast_fce)
        self._corpus.append({"functions": fce, "docstrings": docs})
        return self.generic_visit(node)

    @property
    def corpus(self):
        """
        Create google-style docstring for the function:

        @property
        def corpus(self):
            return pd.DataFrame(self._corpus, columns=['functions', 'docstrings'])
        """
        return pd.DataFrame(self._corpus, columns=["functions", "docstrings"])


def load_corpus(corpus_path: pathlib.Path) -> pd.DataFrame:
    """
    Load a corpus from the specified path and return a pandas DataFrame containing the extracted documentation samples.

    Parameters:
    corpus_path (pathlib.Path): The path to the corpus file to be loaded.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the extracted documentation samples from the corpus.
    """
    raw_code = corpus_path.read_text(encoding="utf-8")
    logging.debug(f"Loaded corpus data of length: {len(raw_code)}")
    tree = ast.parse(raw_code)
    extract_docstring = ExtractDocstrings()
    extract_docstring.visit(tree)
    corpus = extract_docstring.corpus
    logging.info(f"Loaded {len(corpus)} samples from corpus")
    return corpus


def save_annotations(data: pd.DataFrame, output: pathlib.Path):
    """
    Save annotations to a text file.

    Args:
        data (pd.DataFrame): DataFrame containing predictions and functions.
        output (pathlib.Path): Path to the output text file.

    Returns:
        None
    """
    annotated = [
        f'''"""{row['predictions']}"""\n{row['functions']}'''
        for (_, row) in data.iterrows()
    ]
    txt_data = "\n\n\n".join(annotated)
    output.write_text(txt_data, encoding="utf-8")
