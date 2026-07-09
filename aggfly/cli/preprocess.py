"""Resolve a config's ``preprocess`` into a callable applied to the DataArray.

Three mechanisms, in priority order:

1. **Named builtin** — a small registry of common climate conversions
   (``kelvin_to_celsius`` etc.). Safe, no code exec.
2. **Safe arithmetic expression** — a string in the single variable ``x``
   (``"x - 273.15"``, ``"(x - 32) * 5 / 9"``, ``"x ** 2"``). Parsed with ``ast``
   and evaluated against a strict node allowlist — no calls, attributes,
   subscripts, or names other than ``x``. No ``eval`` of arbitrary code.
3. **File escape hatch** (``preprocess_from: path/to/file.py:func``) — imports a
   user function. This runs arbitrary code and is documented as trusted-only.

The resolved callable takes the raw ``xarray.DataArray`` and returns a
transformed one; arithmetic operators dispatch to xarray/numpy as usual.
"""

import ast
import importlib.util
import operator
import os

# ---- named builtins ----------------------------------------------------------
BUILTINS = {
    "identity": lambda x: x,
    "kelvin_to_celsius": lambda x: x - 273.15,
    "celsius_to_kelvin": lambda x: x + 273.15,
    "pa_to_kpa": lambda x: x / 1000.0,
    "m_to_mm": lambda x: x * 1000.0,
}


class PreprocessError(Exception):
    """Raised when a ``preprocess``/``preprocess_from`` spec cannot be resolved."""


# ---- safe arithmetic expression evaluator ------------------------------------
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _validate_node(node):
    """Raise PreprocessError if ``node`` is outside the arithmetic-on-``x`` allowlist."""
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise PreprocessError(f"operator {type(node.op).__name__} is not allowed")
        _validate_node(node.left)
        _validate_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise PreprocessError(f"unary {type(node.op).__name__} is not allowed")
        _validate_node(node.operand)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise PreprocessError(
                f"only numeric constants are allowed, got {node.value!r}"
            )
    elif isinstance(node, ast.Name):
        if node.id != "x":
            raise PreprocessError(
                f"only the variable 'x' is allowed, got {node.id!r}"
            )
    else:
        raise PreprocessError(
            f"expression element {type(node).__name__} is not allowed "
            "(only arithmetic on 'x' and numbers)"
        )


def _eval_node(node, x):
    """Recursively evaluate a pre-validated arithmetic AST with ``x`` bound."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, x)
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](
            _eval_node(node.left, x), _eval_node(node.right, x)
        )
    if isinstance(node, ast.UnaryOp):
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, x))
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):  # guaranteed to be 'x' by validation
        return x
    raise PreprocessError(f"unexpected node {type(node).__name__}")  # pragma: no cover


def _references_x(tree):
    return any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(tree))


def compile_expression(expr):
    """Compile a safe arithmetic-in-``x`` expression string into a callable."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise PreprocessError(f"could not parse expression {expr!r}: {e.msg}")
    _validate_node(tree)
    if not _references_x(tree):
        raise PreprocessError(
            f"expression {expr!r} must use the variable 'x' "
            "(e.g. 'x - 273.15')"
        )
    return lambda x: _eval_node(tree, x)


# ---- file escape hatch -------------------------------------------------------
def load_from_file(spec):
    """Load ``func`` from ``path/to/file.py:func`` (runs arbitrary user code)."""
    if ":" not in spec:
        raise PreprocessError(
            f"preprocess_from must be 'path/to/file.py:function', got {spec!r}"
        )
    path, func_name = spec.rsplit(":", 1)
    if not os.path.exists(path):
        raise PreprocessError(f"preprocess_from file not found: {path}")
    module_spec = importlib.util.spec_from_file_location("aggfly_user_preprocess", path)
    if module_spec is None or module_spec.loader is None:
        raise PreprocessError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as e:
        raise PreprocessError(f"error importing {path}: {e}")
    func = getattr(module, func_name, None)
    if func is None:
        raise PreprocessError(f"function {func_name!r} not found in {path}")
    if not callable(func):
        raise PreprocessError(f"{func_name!r} in {path} is not callable")
    return func


# ---- public entry point ------------------------------------------------------
def resolve(preprocess=None, preprocess_from=None):
    """Resolve a preprocess spec to a callable (or None if neither is set)."""
    if preprocess is not None and preprocess_from is not None:
        raise PreprocessError(
            "set at most one of 'preprocess' and 'preprocess_from'"
        )
    if preprocess_from is not None:
        return load_from_file(preprocess_from)
    if preprocess is None:
        return None
    # a bare registered name, otherwise a safe arithmetic expression
    if isinstance(preprocess, str) and preprocess in BUILTINS:
        return BUILTINS[preprocess]
    if not isinstance(preprocess, str):
        raise PreprocessError(
            f"preprocess must be a builtin name or an expression string, "
            f"got {type(preprocess).__name__}"
        )
    try:
        return compile_expression(preprocess)
    except PreprocessError as e:
        # If it looked like a plain identifier, the user probably meant a builtin.
        if preprocess.isidentifier():
            raise PreprocessError(
                f"unknown preprocess {preprocess!r}: not a builtin "
                f"({', '.join(sorted(BUILTINS))}) and not a valid expression"
            )
        raise e


def resolve_from_config(config):
    """Resolve the preprocess for a RunConfig (convenience for validate/run)."""
    return resolve(config.preprocess, config.preprocess_from)
