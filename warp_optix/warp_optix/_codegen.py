"""OptiX kernel-type helpers.

``OptixKernelType`` was extracted from ``warp/_src/context.py`` and lives here
now. The companion ``optix_kernel`` decorator wraps ``@wp.kernel(...)`` with the
right ``entry_template`` so warp's generic codegen emits an OptiX entry point.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import enum

class OptixKernelType(enum.Enum):
    RAYGEN = "__raygen__"
    MISS = "__miss__"
    CLOSEST_HIT = "__closesthit__"
    ANY_HIT = "__anyhit__"
    INTERSECTION = "__intersection__"



# Template passed to ``@wp.kernel(entry_template=...)`` for OptiX entry points.
# Substitutions ``{entry_name}``, ``{forward_args}``, ``{forward_body}`` and
# ``{global_params_alias}`` are filled in by warp's generic codegen path. The
# ``{prefix}`` slot is filled in by ``optix_template_for`` below using one of
# the OptixKernelType values.
_OPTIX_KERNEL_TEMPLATE_BASE = """

{line_directive}extern "C" __global__ void {prefix}{entry_name}(
    {forward_args})
{{
{global_params_alias}{forward_body}{line_directive}}}

"""


def optix_template_for(kind: "OptixKernelType") -> str:
    """Produce a warp ``entry_template`` string for the given OptiX program kind."""
    return _OPTIX_KERNEL_TEMPLATE_BASE.replace("{prefix}", kind.value)


F = TypeVar("F", bound=Callable[..., Any])


def optix_kernel(kind: "OptixKernelType", **kernel_kwargs: Any) -> Callable[[F], Any]:
    """Decorator: register a Warp kernel as an OptiX entry point of ``kind``.

    Equivalent to ``@wp.kernel(entry_template=optix_template_for(kind), ...)``.
    The decorated function is dispatched by the OptiX pipeline, not by
    ``wp.launch``. The ``kind`` is stashed on the resulting kernel so
    :func:`warp_optix.get_entry_name` can reconstruct the full OptiX entry
    name (e.g. ``__raygen__<mangled>``).
    """
    import warp as wp  # noqa: PLC0415

    def _wrap(fn: F) -> Any:
        k = wp.kernel(entry_template=optix_template_for(kind), **kernel_kwargs)(fn)
        k.options["optix_kernel_type"] = kind
        return k

    return _wrap
