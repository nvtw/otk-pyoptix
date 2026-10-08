"""OptiX kernel-type helpers.

``OptixKernelType`` maps supported OptiX program kinds to their entry-name
prefix. The companion :func:`optix_kernel` decorator assigns the full kernel
name, selects Warp's external constant-params ABI, and stores the kind for
runtime validation.
"""

from __future__ import annotations

import enum
import re
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


def _qualified_kernel_name(fn: Callable[..., Any]) -> str:
    """Return Warp's stable identifier spelling for a Python function."""
    return re.sub("[^0-9a-zA-Z_]+", "", fn.__qualname__.replace(".", "__"))


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
    from warp_optix._compat import (  # noqa: PLC0415
        create_external_kernel,
        has_public_addon_hooks,
    )

    def _wrap(fn: F) -> Any:
        kwargs = dict(kernel_kwargs)

        # OptiX entry programs don't have a meaningful backward pass. We set
        # this on the addon side so Warp's @wp.kernel stays agnostic.
        kwargs.setdefault("enable_backward", False)
        kwargs.setdefault("entry_point_abi", "external_constant_params")
        kernel_name = kwargs.pop("name", _qualified_kernel_name(fn))

        full_name = f"{kind.value}{kernel_name}"
        if has_public_addon_hooks(wp):
            k = wp.kernel(name=full_name, **kwargs)(fn)
        else:
            entry_point_abi = kwargs.pop("entry_point_abi")
            k = create_external_kernel(
                wp,
                fn,
                name=full_name,
                entry_point_abi=entry_point_abi,
                kernel_kwargs=kwargs,
            )
        k.options["optix_kernel_type"] = kind
        return k

    return _wrap
