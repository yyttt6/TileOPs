"""Element-wise logical ops (output bool)."""


from ._base import BinaryOp, UnaryOp


class LogicalAndFwdOp(BinaryOp):
    """Element-wise logical AND with broadcast using non-zero truthiness."""

    _op_name = "logical_and"


class LogicalOrFwdOp(BinaryOp):
    """Element-wise logical OR with broadcast using non-zero truthiness."""

    _op_name = "logical_or"


class LogicalNotFwdOp(UnaryOp):
    """Element-wise logical NOT with bool output."""

    _op_name = "logical_not"
