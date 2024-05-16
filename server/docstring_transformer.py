import ast
from ast import AST, AsyncFunctionDef, ClassDef, Constant, Expr, FunctionDef, Module
import black
from model_loader import ModelI


def get_node_offset(node: ast.stmt) -> int:
    """
    Return the column offset of the first column in a statement node.

        Args:
            node (ast.stmt): The statement node to extract the offset from.

        Returns:
            int: The column offset of the first column in the statement node.
    """
    return node.body[0].col_offset


def indent_string(string: str, indent: int) -> str:
    """
    Indent a string by a specified number of spaces.

    Args:
        string (str): The input string to be indented.
        indent (int): The number of spaces to indent the string by.

    Returns:
        str: The indented string.
    """
    return string.replace("\n", "\n" + " " * indent)


def set_docstring(node: ast.stmt, docstring: str, overwrite=False):
    """
    Set the docstring for a given node in the AST.

    Args:
        node (ast.stmt): The node to set the docstring for.
        docstring (str): The new docstring to set for the node.
        overwrite (bool, optional): Whether to overwrite the existing docstring. Defaults to False.

    Raises:
        TypeError: If the node is not an instance of AsyncFunctionDef, FunctionDef, ClassDef, or Module.

    Returns:
        None.
    """
    if not isinstance(node, (AsyncFunctionDef, FunctionDef, ClassDef, Module)):
        raise TypeError("%r can't have docstrings" % node.__class__.__name__)
    if not (node.body and isinstance(node.body[0], Expr)):
        offset = get_node_offset(node)
        node.body.insert(
            0,
            Expr(
                value=Constant(
                    value=indent_string("\n" + docstring + "\n", offset),
                    col_offset=offset,
                    end_col_offset=offset + 3,
                ),
                col_offset=offset,
                end_col_offset=offset + 3,
            ),
        )
        return
    if overwrite:
        leaf = node.body[0].value
        if isinstance(node, Constant):
            leaf.s = docstring
        elif isinstance(node, Constant) and isinstance(node.value, str):
            leaf.value = docstring


class DocstringAdder(ast.NodeTransformer):
    def __init__(self, docstring_generator: ModelI, overwrite: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docstring_generator = docstring_generator
        self.overwrite_docstrings = overwrite

    def visit_FunctionDef(self, node: ast.stmt) -> AST:
        """
        Visit a FunctionDef node and generate docstring for it.

        Args:
            self: The current instance of the class.
            node (ast.stmt): The FunctionDef node to be visited.

        Returns:
            AST: The modified AST after generating the docstring for the node.
        """
        fce_code = ast.unparse(node)
        docstring = self.docstring_generator.generate(fce_code)
        set_docstring(node, docstring, self.overwrite_docstrings)
        return self.generic_visit(node)


def annotate_code(
    code: str, docstring_generator: ModelI, overwrite_docstrings: bool = False
) -> str:
    """
    Annotates a code block with documentation strings.

    Args:
        code (str): The code block to be annotated.
        docstring_generator (ModelI): A generator that generates documentation strings.
        overwrite_docstrings (bool, optional): Whether to overwrite existing documentation strings. Defaults to False.

    Returns:
        str: The annotated code block with documentation strings added.
    """
    tree = ast.parse(code)
    new_tree = DocstringAdder(docstring_generator, overwrite_docstrings).visit(tree)
    ast.fix_missing_locations(new_tree)
    generated_code = ast.unparse(new_tree)
    return black.format_str(generated_code, mode=black.Mode())


def generate_docstring(code: str, docstring_generator: ModelI) -> str:
    """
    Generate docstring for a given code using the provided docstring generator.

    Args:
        code (str): The code to generate the docstring for.
        docstring_generator (ModelI): The docstring generator used to generate the docstring.

    Returns:
        str: The generated docstring for the code.
    """
    return docstring_generator.generate(code)
