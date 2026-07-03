"""OptiX kernel-type helpers.

``OptixKernelType`` maps supported OptiX program kinds to the name prefix
that Warp's codegen attaches to the generated CUDA entry point. The
companion :func:`optix_kernel` decorator selects Warp's external constant
params ABI and stores the kind for runtime entry-name lookup.
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
    EXCEPTION = "__exception__"
    DIRECT_CALLABLE = "__direct_callable__"
    CONTINUATION_CALLABLE = "__continuation_callable__"


F = TypeVar("F", bound=Callable[..., Any])


def optix_kernel(kind: "OptixKernelType", **kernel_kwargs: Any) -> Callable[[F], Any]:
    """Decorator: register a Warp kernel as an OptiX entry point of ``kind``.

    Equivalent to ``@wp.kernel(entry_prefix=kind.value,
    entry_point_abi="external_constant_params", enable_backward=False, ...)``.
    The single Warp struct argument is bound to OptiX launch params ``params``.
    The kind is stored so
    :func:`warp_optix.get_entry_name` can reconstruct the full entry name
    (e.g. ``__raygen__<mangled>``). The decorated function is dispatched by
    the OptiX pipeline, not by ``wp.launch``.
    """
    import warp as wp  # noqa: PLC0415

    def _wrap(fn: F) -> Any:
        kwargs = dict(kernel_kwargs)

        # OptiX entry programs don't have a meaningful backward pass. We set
        # this on the addon side so Warp's @wp.kernel stays agnostic.
        kwargs.setdefault("enable_backward", False)
        kwargs.setdefault("entry_point_abi", "external_constant_params")

        k = wp.kernel(entry_prefix=kind.value, **kwargs)(fn)
        k.options["optix_kernel_type"] = kind
        return k

    return _wrap
