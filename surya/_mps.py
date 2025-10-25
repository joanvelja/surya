from __future__ import annotations

import logging
from typing import Final

import torch
from torch import Tensor

LOGGER: Final = logging.getLogger(__name__)

_ORIGINAL_ISIN = torch.isin
_PATCHED: Final[dict[str, bool]] = {"isin": False}


def configure_mps_support() -> None:
    """
    Apply runtime shims for operators that remain unavailable on the MPS backend.

    Torch 2.9 expands MPS coverage but `torch.isin` can still raise on some
    builds. Mirror the eager implementation so Surya stays on-device without
    enabling the global CPU fallback.
    """
    if _PATCHED["isin"] or not torch.backends.mps.is_available():
        return

    device = torch.device("mps")

    try:
        _ORIGINAL_ISIN(
            torch.tensor([0], device=device),
            torch.tensor([0], device=device),
        )
        _PATCHED["isin"] = True
        return
    except (RuntimeError, NotImplementedError):
        LOGGER.info(
            "torch.isin is not implemented on MPS; installing tensor fallback "
            "to avoid CPU execution."
        )

    def _isin_fallback(
        elements: Tensor,
        test_elements: Tensor,
        *,
        assume_unique: bool = False,
        invert: bool = False,
    ) -> Tensor:
        if elements.device.type != "mps":
            return _ORIGINAL_ISIN(
                elements,
                test_elements,
                assume_unique=assume_unique,
                invert=invert,
            )

        if elements.numel() == 0:
            result = torch.zeros(
                elements.shape,
                dtype=torch.bool,
                device=elements.device,
            )
            return torch.logical_not(result) if invert else result

        if test_elements.numel() == 0:
            result = torch.zeros(
                elements.shape,
                dtype=torch.bool,
                device=elements.device,
            )
            return result if not invert else torch.logical_not(result)

        target = test_elements.to(device=elements.device)
        if not assume_unique:
            target = torch.unique(target)

        elems = elements.reshape(-1, 1)
        target = target.reshape(1, -1)

        matches = torch.eq(elems, target).any(dim=1)
        if invert:
            matches = torch.logical_not(matches)

        return matches.reshape(elements.shape)

    torch.isin = _isin_fallback  # type: ignore[assignment]
    Tensor.isin = _isin_fallback  # type: ignore[assignment]
    _PATCHED["isin"] = True
