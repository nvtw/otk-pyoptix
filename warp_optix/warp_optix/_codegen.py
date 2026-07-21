"""OptiX kernel-type helpers.

``OptixKernelType`` maps supported OptiX program kinds to their entry-name
prefix. The companion :func:`optix_kernel` decorator assigns the full kernel
name, selects Warp's external constant-params ABI, and stores the kind for
runtime validation.
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

    Equivalent to ``@wp.kernel(name=kind.value + name,
    entry_point_abi="external_constant_params", enable_backward=False, ...)``.
    The single Warp struct argument is bound to OptiX launch params ``params``.
    An optional ``name`` keyword overrides the function-derived portion of the
    name; the OptiX program-kind prefix is always prepended. The decorated
    function is dispatched by the OptiX pipeline, not by ``wp.launch``.
    """
    import warp as wp  # noqa: PLC0415

    def _wrap(fn: F) -> Any:
        kwargs = dict(kernel_kwargs)

        # OptiX entry programs don't have a meaningful backward pass. We set
        # this on the addon side so Warp's @wp.kernel stays agnostic.
        kwargs.setdefault("enable_backward", False)
        kwargs.setdefault("entry_point_abi", "external_constant_params")
        kernel_name = kwargs.pop("name", wp._src.codegen.make_full_qualified_name(fn))

        k = wp.kernel(name=f"{kind.value}{kernel_name}", **kwargs)(fn)
        k.options["optix_kernel_type"] = kind
        return k

    return _wrap
