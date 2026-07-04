"""Device resolution with a graceful CPU fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ._import_utils import require_torch

if TYPE_CHECKING:
    import torch

_logger = logging.getLogger("evozero")
_warned = False


def resolve_device(requested: str = "auto") -> "torch.device":
    """Resolve a device string to a concrete :class:`torch.device`.

    Parameters
    ----------
    requested : str, default="auto"
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``"mps"``.
        ``"auto"`` picks the CUDA device with the most free memory, else MPS,
        else CPU (emitting a single warning).

    Returns
    -------
    torch.device
    """
    require_torch()
    import torch

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        return torch.device(f"cuda:{best}")

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    global _warned
    if not _warned:
        _logger.warning("No GPU accelerator found; evozero falls back to CPU.")
        _warned = True
    return torch.device("cpu")
