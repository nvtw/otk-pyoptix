"""OptiX kernel-type helpers.

``OptixKernelType`` maps the five OptiX program kinds to the name prefix
that warp's codegen attaches to the generated CUDA entry point. The
companion :func:`optix_kernel` decorator is a thin wrapper around
``@wp.kernel(entry_prefix=...)`` that also stashes the kind on the kernel
so the runtime can reconstruct the full entry name.
"""

from __future__ import annotations

import enum
from typing import Any, Callable, TypeVar


class OptixKernelType(enum.Enum):
    RAYGEN = "__raygen__"
    MISS = "__miss__"
    CLOSEST_HIT = "__closesthit__"
    ANY_HIT = "__anyhit__"
    INTERSECTION = "__intersection__"


F = TypeVar("F", bound=Callable[..., Any])


def optix_kernel(kind: "OptixKernelType", **kernel_kwargs: Any) -> Callable[[F], Any]:
    """Decorator: register a Warp kernel as an OptiX entry point of ``kind``.

    Equivalent to ``@wp.kernel(entry_prefix=kind.value, enable_backward=False, ...)``
    plus a stash of ``kind`` on the resulting kernel so
    :func:`warp_optix.get_entry_name` can reconstruct the full entry name
    (e.g. ``__raygen__<mangled>``). The decorated function is dispatched by
    the OptiX pipeline, not by ``wp.launch``.
    """
    import warp as wp  # noqa: PLC0415

    # OptiX entry programs don't have a meaningful backward pass. We set
    # this on the addon side so warp's @wp.kernel stays agnostic.
    kernel_kwargs.setdefault("enable_backward", False)

    def _wrap(fn: F) -> Any:
        k = wp.kernel(entry_prefix=kind.value, **kernel_kwargs)(fn)
        k.options["optix_kernel_type"] = kind
        return k

    return _wrap
