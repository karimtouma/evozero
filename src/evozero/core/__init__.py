"""Internal GPU engines (import torch).

Everything in this subpackage may import :mod:`torch` at module top level.
The public facades in :mod:`evozero.sr` and :mod:`evozero.automlzero` import
from here lazily, so ``import evozero`` stays torch-free.
"""
