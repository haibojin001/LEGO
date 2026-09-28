"""Tiny arithmetic package used by the LEGO smoke test."""
from .ops import add, mul
from .stats import mean, total

__all__ = ["add", "mul", "mean", "total"]
