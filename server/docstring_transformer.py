import ast
from ast import AST, AsyncFunctionDef, ClassDef, Constant, Expr, FunctionDef, Module

import black
from model_loader import ModelI


def get_node_offset(node: ast.stmt) -> int:
    return node.body[0].col_offset


def indent_string(string: str, indent: int) -> str:
    return string.replace("\n", "\n" + " " * indent)


def set_docstring(node: ast.stmt, docstring: str, overwrite=False):
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
        fce_code = ast.unparse(node)
        docstring = self.docstring_generator.generate(fce_code)
        set_docstring(node, docstring, self.overwrite_docstrings)
        return self.generic_visit(node)


def annotate_code(code: str, docstring_generator: ModelI, overwrite_docstrings: bool = False) -> str:
    tree = ast.parse(code)
    new_tree = DocstringAdder(docstring_generator, overwrite_docstrings).visit(tree)
    ast.fix_missing_locations(new_tree)
    generated_code = ast.unparse(new_tree)
    return black.format_str(generated_code, mode=black.Mode())
