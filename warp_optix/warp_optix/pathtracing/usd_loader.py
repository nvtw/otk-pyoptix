"""OpenUSD scene loader for the path tracer.

OpenUSD is imported lazily so USD remains an optional feature.  Geometry and
materials are translated directly into the path tracer's glTF-style PBR data;
no intermediate glTF file is written.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .scene import Scene

logger = logging.getLogger(__name__)
_MIN_TEXTURE_MEMORY_BUDGET_BYTES = 1 << 30
_TEXTURE_VRAM_RESERVE_BYTES = 4 << 30


def _import_usd():
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade
    except ImportError as exc:
        raise ImportError(
            "USD loading requires the optional OpenUSD Python bindings. "
            "Install warp_optix with the 'usd' extra (pip install "
            '-e "warp_optix[pathtracing,usd]").'
        ) from exc
    return Sdf, Usd, UsdGeom, UsdLux, UsdShade


def _as_float_tuple(
    value: Any, size: int, default: tuple[float, ...]
) -> tuple[float, ...]:
    if value is None:
        return default
    try:
        values = tuple(float(value[i]) for i in range(size))
    except (IndexError, TypeError):
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return default
        values = (scalar,) * size
    return values


def _ambient_light_from_custom_layer_data(
    custom_layer_data: Any,
) -> tuple[float, float, float]:
    """Read Omniverse sceneDb ambient irradiance from USD layer metadata."""
    data = dict(custom_layer_data or {})
    render_settings = dict(data.get("renderSettings", {}) or {})
    color = render_settings.get("rtx:sceneDb:ambientLightColor")
    intensity = float(render_settings.get("rtx:sceneDb:ambientLightIntensity", 0.0))
    if color is None or intensity <= 0.0:
        return (0.0, 0.0, 0.0)
    rgb = _as_float_tuple(color, 3, (0.0, 0.0, 0.0))
    return tuple(max(0.0, component * intensity) for component in rgb)


def _ellipsoid_surface_area(radii: tuple[float, float, float]) -> float:
    """Approximate ellipsoid area for a transformed sphere light."""
    a, b, c = (max(abs(float(value)), 1.0e-9) for value in radii)
    power = 1.6075
    return (
        4.0
        * math.pi
        * ((a**power * b**power + a**power * c**power + b**power * c**power) / 3.0)
        ** (1.0 / power)
    )


def _sphere_light_proxy_radiance(
    intensity: float,
    normalize: bool,
    authored_radius: float,
    world_transform: np.ndarray,
    proxy_radius: float,
) -> float:
    """Preserve authored sphere-light power when changing its render radius."""
    linear = np.asarray(world_transform, dtype=np.float64)[:3, :3]
    radii = tuple(
        authored_radius * float(np.linalg.norm(linear[:, axis])) for axis in range(3)
    )
    authored_area = _ellipsoid_surface_area(radii)
    proxy_area = 4.0 * math.pi * proxy_radius * proxy_radius
    return (
        intensity / proxy_area if normalize else intensity * authored_area / proxy_area
    )


def _material_inputs(material, shader) -> dict[str, Any]:
    """Return authored material values, with shader defaults as a fallback."""
    values: dict[str, Any] = {}
    if shader:
        values.update(
            {
                str(i.GetBaseName()): i.Get()
                for i in shader.GetInputs()
                if i.Get() is not None
            }
        )
    if material:
        values.update(
            {
                str(i.GetBaseName()): i.Get()
                for i in material.GetInputs()
                if i.Get() is not None
            }
        )
    return values


def _surface_shader(material, UsdShade):
    for context in ("", "universal", "mdl"):
        try:
            source = material.ComputeSurfaceSource(context)
        except Exception:
            continue
        if source and source[0]:
            return UsdShade.Shader(source[0].GetPrim())
    for child in material.GetPrim().GetChildren():
        if child.IsA(UsdShade.Shader):
            return UsdShade.Shader(child)
    return None


def _bound_material(prim, UsdShade):
    """Resolve direct bindings even when MaterialBindingAPI was not applied."""
    if not prim:
        return None
    if not prim.HasAPI(UsdShade.MaterialBindingAPI):
        relationships = sorted(
            (
                relationship
                for relationship in prim.GetRelationships()
                if relationship.GetName().startswith("material:binding")
            ),
            key=lambda relationship: relationship.GetName() != "material:binding",
        )
        for relationship in relationships:
            targets = relationship.GetTargets()
            if targets:
                material = UsdShade.Material(prim.GetStage().GetPrimAtPath(targets[0]))
                if material:
                    return material
    return UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]


def _material_bind_subsets(prim, UsdGeom, UsdShade):
    """Return material subsets without requiring an applied binding schema."""
    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        return UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    return [
        UsdGeom.Subset(child)
        for child in prim.GetChildren()
        if child.IsA(UsdGeom.Subset)
        and str(UsdGeom.Subset(child).GetFamilyNameAttr().Get()) == "materialBind"
    ]


def _shader_source_identity(shader) -> str:
    """Return a stable shader identity for id- and source-asset-based networks."""
    if not shader:
        return ""
    shader_id = str(shader.GetIdAttr().Get() or "")
    if shader_id:
        return shader_id
    for context in ("mdl", "universal", ""):
        try:
            source_asset = shader.GetSourceAsset(context)
            sub_identifier = shader.GetSourceAssetSubIdentifier(context)
        except Exception:
            continue
        asset_path = _asset_path(source_asset)
        if asset_path or sub_identifier:
            return f"{asset_path or ''}:{sub_identifier or ''}"
    return ""


def _asset_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value) if value else None
    resolved = getattr(value, "resolvedPath", "")
    authored = getattr(value, "path", "") or getattr(value, "authoredPath", "")
    path = str(resolved or authored)
    return Path(path) if path else None


def _texture_candidate_exists(path: Path) -> bool:
    if "<UDIM>" not in str(path):
        return path.is_file()
    return any(path.parent.glob(path.name.replace("<UDIM>", "*")))


_MDL_TEXTURE_INPUT_ALIASES = {
    "Num1_BaseColor": "diffuse_texture",
    "Num1_BaseColor_1": "diffuse_texture",
    "Num2_Normal": "normalmap_texture",
    "Num2_Normal_1": "normalmap_texture",
    "Num3_Mask": "ORM_texture",
    "Num3_Mask_1": "ORM_texture",
    "Num4_Emissive": "emissive_texture",
    "diffuse_texture": "diffuse_texture",
    "normalmap_texture": "normalmap_texture",
    "ORM_texture": "ORM_texture",
    "reflectionroughness_texture": "reflectionroughness_texture",
    "metallic_texture": "metallic_texture",
}


def _mdl_texture_semantic(name: str) -> str | None:
    """Infer a supported PBR input from a common MDL texture parameter name."""
    exact = _MDL_TEXTURE_INPUT_ALIASES.get(name)
    if exact is not None:
        return exact
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if "opacity" in normalized or "transmission" in normalized:
        return None
    if "normal" in normalized or "bump" in normalized:
        return "normalmap_texture"
    if "emiss" in normalized or "emission" in normalized:
        return "emissive_texture"
    if (
        normalized.startswith("orm")
        or "ormtexture" in normalized
        or "ormmap" in normalized
        or "mask" in normalized
    ):
        return "ORM_texture"
    if "rough" in normalized:
        return "reflectionroughness_texture"
    if "metal" in normalized:
        return "metallic_texture"
    if any(token in normalized for token in ("basecolor", "albedo", "diffuse")):
        return "diffuse_texture"
    return None


@lru_cache(maxsize=None)
def _parse_mdl_texture_inputs(mdl_path: Path) -> tuple[tuple[str, Path], ...]:
    """Parse texture defaults from a collected MDL source asset."""
    if not mdl_path.is_file():
        return ()
    source = mdl_path.read_text(errors="replace")
    declarations = dict(
        re.findall(
            r'(?:uniform\s+)?texture_2d\s+(\w+)\s*=\s*texture_2d\(\s*"([^"]+)"',
            source,
        )
    )
    declarations.update(
        re.findall(
            r'\b(\w+)\s*:\s*texture_2d\(\s*"([^"]+)"',
            source,
        )
    )
    values = {}
    for declaration, relative_path in declarations.items():
        input_name = _mdl_texture_semantic(declaration)
        if input_name is None or input_name in values:
            continue
        texture_path = (mdl_path.parent / relative_path).resolve()
        if _texture_candidate_exists(texture_path):
            values[input_name] = texture_path
    return tuple(values.items())


def _mdl_texture_inputs(shader) -> dict[str, Path]:
    """Return supported PBR texture inputs from an MDL source asset."""
    if not shader:
        return {}
    try:
        source_asset = shader.GetSourceAsset("mdl")
    except Exception:
        return {}
    mdl_path = _asset_path(source_asset)
    if mdl_path is None:
        return {}
    return dict(_parse_mdl_texture_inputs(mdl_path))


def _normalize_texture_inputs(values: dict[str, Any]) -> None:
    """Add canonical aliases for authored texture inputs with common names."""
    for name, value in tuple(values.items()):
        semantic = _mdl_texture_semantic(name)
        if semantic is not None and _asset_path(value) is not None:
            values.setdefault(semantic, value)


def _rebase_missing_texture(stage_path: Path, texture_path: Path) -> Path:
    """Rebase a broken texture asset path to a stage-local textures directory."""
    resolved = (
        texture_path if texture_path.is_absolute() else stage_path.parent / texture_path
    )
    resolved = resolved.resolve()
    if _texture_candidate_exists(resolved):
        return resolved

    parts = resolved.parts
    lowered = tuple(part.lower() for part in parts)
    if "textures" in lowered:
        marker = lowered.index("textures")
        candidate = stage_path.parent.joinpath(*parts[marker:]).resolve()
        if _texture_candidate_exists(candidate):
            return candidate
    return resolved


def _dome_texture(prim, UsdLux) -> Path | None:
    """Return a composed DomeLight texture, including legacy USD schemas."""
    if not prim.IsA(UsdLux.DomeLight):
        return None
    # Marbles was authored with the pre-input-namespace attributes. Current
    # OpenUSD's schema accessor only sees inputs:texture:file, so accept both.
    for name in ("inputs:texture:file", "texture:file"):
        attr = prim.GetAttribute(name)
        texture_path = _asset_path(attr.Get()) if attr else None
        if texture_path:
            return texture_path
    return None


def _decode_texture(path: Path, max_size: int | None = None) -> np.ndarray:
    # Pillow is already part of the pathtracing extra and supports the BC7 DDS
    # textures used by NVIDIA's Marbles sample.  Using it directly also avoids
    # requiring a separate DDS package/plugin.
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        if max_size is not None and max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        arr = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    # USD/MDL texture coordinates use a lower-left image origin while decoded
    # rows and the software sampler use a top-left origin. Keep the UV/tangent
    # basis intact and convert the image storage once here.
    arr = np.flip(arr, axis=0)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _decode_udim(
    tiles: tuple[tuple[int, Path], ...], max_size: int | None = None
) -> np.ndarray:
    """Decode UDIM tiles into a grid atlas using the standard 10-column layout."""
    offsets = [(number - 1001) % 10 for number, _ in tiles]
    rows = [(number - 1001) // 10 for number, _ in tiles]
    column_count = max(offsets) + 1
    row_count = max(rows) + 1
    # The public cap applies to each source texture, just as it does for a
    # non-UDIM image. Dividing it by the tile count made RacerX's nominal
    # 512-pixel setting only ~170 pixels per tile.
    per_tile_max = max_size
    images = [_decode_texture(path, max_size=per_tile_max) for _, path in tiles]
    height = max(tile.shape[0] for tile in images)
    width = max(tile.shape[1] for tile in images)
    if any(tile.shape[:2] != (height, width) for tile in images):
        from PIL import Image  # noqa: PLC0415

        resized = []
        for tile in images:
            resized.append(
                np.asarray(
                    Image.fromarray(tile, mode="RGBA").resize(
                        (width, height), Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8,
                )
            )
        images = resized
    atlas = np.zeros((height * row_count, width * column_count, 4), dtype=np.uint8)
    for (number, _), image in zip(tiles, images, strict=True):
        column = (number - 1001) % 10
        row = (number - 1001) // 10
        # Tile images are already vertically flipped; place increasing UDIM V
        # upward in UV space, which is downward in the stored image.
        storage_row = row_count - 1 - row
        atlas[
            storage_row * height : (storage_row + 1) * height,
            column * width : (column + 1) * width,
        ] = image
    return np.ascontiguousarray(atlas, dtype=np.uint8)


def _decode_packed_orm(spec: tuple, max_size: int | None = None) -> np.ndarray:
    """Build glTF R=AO/G=roughness/B=metallic from separate USD maps."""
    (
        _,
        rough_path,
        metal_path,
        rough_constant,
        metal_constant,
        rough_influence,
        metal_influence,
    ) = spec

    def decode(source):
        if not source:
            return None
        if isinstance(source, tuple) and source[0] == "udim":
            return _decode_udim(source[1], max_size=max_size)
        return _decode_texture(source, max_size=max_size)

    rough = decode(rough_path)
    metal = decode(metal_path)
    reference = rough if rough is not None else metal
    height, width = reference.shape[:2]

    def channel_or_constant(image, value):
        if image is None:
            return np.full((height, width), value, dtype=np.float32)
        if image.shape[:2] != (height, width):
            from PIL import Image  # noqa: PLC0415

            image = np.asarray(
                Image.fromarray(image, mode="RGBA").resize(
                    (width, height), Image.Resampling.LANCZOS
                ),
                dtype=np.uint8,
            )
        return image[..., 0].astype(np.float32) * (1.0 / 255.0)

    rough_source = channel_or_constant(rough, rough_constant)
    metal_source = channel_or_constant(metal, metal_constant)
    roughness = (
        rough_constant * (1.0 - rough_influence) + rough_source * rough_influence
    )
    metallic = metal_constant * (1.0 - metal_influence) + metal_source * metal_influence
    packed = np.full((height, width, 4), 255, dtype=np.uint8)
    packed[..., 1] = np.clip(roughness * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    packed[..., 2] = np.clip(metallic * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(packed)


def _fit_textures_to_budget(
    textures: list[np.ndarray], max_bytes: int | None = None
) -> list[np.ndarray]:
    """Proportionally downscale RGBA textures to fit a GPU-memory budget."""
    total_bytes = sum(texture.nbytes for texture in textures)
    if max_bytes is None:
        try:
            import warp as wp  # noqa: PLC0415

            free_bytes = int(wp.get_device("cuda").free_memory)
            # Keep both half of currently free VRAM and 4 GiB available for
            # geometry, OptiX structures, frame buffers, and DLSS-RR resources.
            max_bytes = min(
                free_bytes // 2,
                max(0, free_bytes - _TEXTURE_VRAM_RESERVE_BYTES),
            )
            max_bytes = max(_MIN_TEXTURE_MEMORY_BUDGET_BYTES, max_bytes)
        except Exception:
            max_bytes = _MIN_TEXTURE_MEMORY_BUDGET_BYTES
    if total_bytes <= max_bytes:
        return textures
    if max_bytes <= 0:
        raise ValueError("Texture byte budget must be positive")

    from PIL import Image  # noqa: PLC0415

    # Pixel storage is RGBA8, so reducing both dimensions by this scale reduces
    # aggregate memory quadratically. Leave rounding headroom for small images.
    scale = min(1.0, (max_bytes / total_bytes) ** 0.5 * 0.99)
    resized = []
    for texture in textures:
        height, width = texture.shape[:2]
        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
        if (target_height, target_width) == (height, width):
            resized.append(texture)
            continue
        image = Image.fromarray(texture, mode="RGBA").resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        resized.append(np.ascontiguousarray(image, dtype=np.uint8))

    resized_bytes = sum(texture.nbytes for texture in resized)
    logger.warning(
        "USD textures exceeded the %.2f GiB GPU budget; downscaled %d textures from %.2f GiB to %.2f GiB",
        max_bytes / (1 << 30),
        len(textures),
        total_bytes / (1 << 30),
        resized_bytes / (1 << 30),
    )
    return resized


def _preview_texture(shader_input) -> Path | None:
    """Follow a UsdPreviewSurface input to a UsdUVTexture file input."""
    if not shader_input:
        return None
    try:
        source = shader_input.GetConnectedSource()
    except Exception:
        return None
    if not source or not source[0]:
        return None
    source_shader = source[0]
    file_input = source_shader.GetInput("file")
    return _asset_path(file_input.Get()) if file_input else None


def _material_to_pbr(
    material,
    UsdShade,
    texture_index,
    packed_orm_index=None,
    enable_emissive_materials: bool = True,
) -> dict[str, Any]:
    shader = _surface_shader(material, UsdShade)
    values = _mdl_texture_inputs(shader)
    values.update(_material_inputs(material, shader))
    shader_id = _shader_source_identity(shader)

    _normalize_texture_inputs(values)
    texture_scale = _as_float_tuple(values.get("texture_scale"), 2, (1.0, 1.0))
    texture_offset = _as_float_tuple(values.get("texture_translate"), 2, (0.0, 0.0))
    texture_rotation = np.deg2rad(float(values.get("texture_rotate", 0.0)))
    texture_transform = {
        "scale": texture_scale,
        "offset": texture_offset,
        "rotation": texture_rotation,
    }
    texture_uv_set = int(values.get("uv_space_index", 0))

    def texture(*names: str, srgb: bool = False) -> dict[str, int]:
        path = None
        for name in names:
            path = _asset_path(values.get(name))
            if path:
                break
        if path is None and shader:
            for name in names:
                path = _preview_texture(shader.GetInput(name))
                if path:
                    break
        registered = texture_index(path, srgb) if path else -1
        if isinstance(registered, tuple):
            index, columns, rows = registered
            return {
                "index": index,
                "texCoord": texture_uv_set,
                "transform": texture_transform,
                # UV normalization is mesh-specific. RacerX B3, for example,
                # binds a four-tile material to a chassis whose UVs span only
                # three tiles. Keep the atlas grid as loader metadata instead
                # of applying one material-wide transform.
                "udim_grid": (columns, rows),
            }
        info = {
            "index": registered,
            "texCoord": texture_uv_set,
            "transform": texture_transform,
        }
        # A material may combine a stitched UDIM atlas (for example albedo)
        # with a single numbered tile (for example emissive.1003).  Record
        # the latter so its UVs can be remapped back from atlas space.
        if path is not None:
            match = re.search(r"\.(1\d{3})(?=\.[^.]+$)", path.name)
            if match:
                info["udim_tile"] = int(match.group(1))
        return info

    def connected_path(*names: str) -> Path | None:
        for name in names:
            path = _asset_path(values.get(name))
            if path:
                return path
        if shader:
            for name in names:
                path = _preview_texture(shader.GetInput(name))
                if path:
                    return path
        return None

    if shader_id == "UsdPreviewSurface":
        base = _as_float_tuple(values.get("diffuseColor"), 3, (0.18, 0.18, 0.18))
        opacity = float(values.get("opacity", 1.0))
        emissive = (
            _as_float_tuple(values.get("emissiveColor"), 3, (0.0, 0.0, 0.0))
            if enable_emissive_materials
            else (0.0, 0.0, 0.0)
        )
        roughness = float(values.get("roughness", 0.5))
        metallic = float(values.get("metallic", 0.0))
        orm_index = -1
        if packed_orm_index is not None:
            orm_index = packed_orm_index(
                connected_path("roughness"),
                connected_path("metallic"),
                roughness,
                metallic,
                1.0,
                1.0,
            )
        has_orm = orm_index >= 0
        orm_texture = {"index": orm_index, "texCoord": 0}
        return {
            "base_color": (*base, opacity),
            "emissive_factor": emissive,
            "metallic": 1.0 if has_orm else metallic,
            "roughness": 1.0 if has_orm else roughness,
            "ior": float(values.get("ior", 1.5)),
            "transmission": 0.0,
            "alpha_mode": "MASK"
            if float(values.get("opacityThreshold", 0.0)) > 0.0
            else ("BLEND" if opacity < 0.999 else "OPAQUE"),
            "alpha_cutoff": float(values.get("opacityThreshold", 0.5)),
            "base_color_texture": texture("diffuseColor", srgb=True),
            "normal_texture": texture("normal"),
            "metallic_roughness_texture": orm_texture,
            "emissive_texture": texture("emissiveColor", srgb=True)
            if enable_emissive_materials
            else {},
            "occlusion_texture": texture("occlusion"),
        }

    # NVIDIA MDL OmniPBR/OmniGlass conventions used by Marbles.
    identity_lower = shader_id.lower()
    is_glass = (
        "omniglass" in identity_lower
        or "glass_color" in values
        or "glass_ior" in values
    )
    diffuse_constant = _as_float_tuple(
        values.get("diffuse_color_constant", values.get("albedo_color")),
        3,
        (0.18, 0.18, 0.18),
    )
    diffuse_tint = _as_float_tuple(values.get("diffuse_tint"), 3, (1.0, 1.0, 1.0))
    base_color_texture = texture("diffuse_texture", srgb=True)
    if is_glass:
        base = _as_float_tuple(values.get("glass_color"), 3, (1.0, 1.0, 1.0))
    elif base_color_texture["index"] >= 0:
        brightness = float(values.get("albedo_brightness", 1.0))
        base = tuple(component * brightness for component in diffuse_tint)
    else:
        base = diffuse_tint if diffuse_tint != (1.0, 1.0, 1.0) else diffuse_constant
    emission_enabled = enable_emissive_materials and (
        bool(values.get("enable_emission", False))
        or "emissive_color_normal" in values
        or "emissive_color_grazing" in values
    )
    emissive = _as_float_tuple(
        values.get("emissive_color", values.get("emissive_color_normal")),
        3,
        (0.0, 0.0, 0.0),
    )
    emissive_scale = (
        float(values.get("emissive_intensity", 0.0)) if emission_enabled else 0.0
    )
    orm_enabled = bool(values.get("enable_ORM_texture", True))
    orm_texture = (
        texture("ORM_texture")
        if orm_enabled
        else {"index": -1, "texCoord": texture_uv_set}
    )
    has_orm = orm_texture["index"] >= 0
    normal_texture = texture("normalmap_texture")
    if normal_texture["index"] >= 0:
        normal_texture["scale"] = float(values.get("bump_factor", 1.0))
    metallic = float(values.get("metallic_constant", 0.0))
    roughness_default = 0.0 if is_glass else 0.5
    roughness = float(
        values.get(
            "frosting_roughness",
            values.get(
                "reflection_roughness_constant",
                values.get("reflection_roughness", roughness_default),
            ),
        )
    )
    if not has_orm and packed_orm_index is not None:
        orm_index = packed_orm_index(
            connected_path(
                "reflectionroughness_texture", "reflection_roughness_texture"
            ),
            connected_path("metallic_texture"),
            roughness,
            metallic,
            float(values.get("reflection_roughness_texture_influence", 1.0)),
            float(values.get("metallic_texture_influence", 1.0)),
        )
        if orm_index >= 0:
            orm_texture = {"index": orm_index, "texCoord": 0}
            has_orm = True
    if has_orm:
        # glTF multiplies the packed B/G channels by these factors. Preserve
        # the authored ORM values unchanged, matching minimaldlssrr.
        metallic = 1.0
        roughness = 1.0
    return {
        "base_color": (*base, 1.0),
        "emissive_factor": tuple(v * emissive_scale for v in emissive),
        "metallic": metallic,
        "roughness": roughness,
        "ior": float(values.get("glass_ior", 1.5)),
        "transmission": 1.0 if is_glass else 0.0,
        "alpha_mode": "BLEND" if is_glass else "OPAQUE",
        "specular": float(values.get("specular_level", 1.0)),
        "thickness": 0.0 if bool(values.get("thin_walled", False)) else 1.0,
        "base_color_add": float(values.get("albedo_add", 0.0)),
        "base_color_desaturation": float(values.get("albedo_desaturation", 0.0)),
        "base_color_texture": base_color_texture,
        "normal_texture": normal_texture,
        # Marbles ORM is already packed as glTF expects: R=AO, G=roughness, B=metallic.
        "metallic_roughness_texture": orm_texture,
        "emissive_texture": texture(
            "emissive_texture", "emissive_color_texture", srgb=True
        )
        if enable_emissive_materials
        else {},
        "occlusion_texture": orm_texture,
    }


def _expanded_attribute(
    values, indices, interpolation: str, point_indices, face_ids, corner_ids, width: int
):
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1, width)
    if interpolation == "faceVarying":
        element = corner_ids
    elif interpolation in ("vertex", "varying"):
        element = point_indices
    elif interpolation == "uniform":
        element = face_ids
    else:
        element = np.zeros(len(corner_ids), dtype=np.int64)
    if indices is not None and len(indices):
        element = np.asarray(indices, dtype=np.int64)[element]
    return arr[element]


def _mesh_arrays(mesh, UsdGeom):
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    point_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not point_indices:
        return None

    counts = np.asarray(counts, dtype=np.int64)
    point_indices = np.asarray(point_indices, dtype=np.int64)
    corner_ids = np.arange(len(point_indices), dtype=np.int64)
    face_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    points_np = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    vertices = points_np[point_indices]

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    normals_primvar = primvars.GetPrimvar("normals")
    normals_value = mesh.GetNormalsAttr().Get()
    normals_indices = None
    normals_interpolation = str(mesh.GetNormalsInterpolation())
    if normals_primvar and normals_primvar.HasValue():
        normals_value = normals_primvar.Get()
        normals_indices = normals_primvar.GetIndices()
        normals_interpolation = str(normals_primvar.GetInterpolation())
    normals = _expanded_attribute(
        normals_value,
        normals_indices,
        normals_interpolation,
        point_indices,
        face_ids,
        corner_ids,
        3,
    )
    if normals is not None:
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(length, 1.0e-20)

    texcoord_types = {
        "texcoord2f[]",
        "texcoord2d[]",
        "texcoord2h[]",
        "float2[]",
        "double2[]",
        "half2[]",
    }
    st = None
    for name in (
        "st",
        "st_0",
        "st0",
        "uv",
        "uv_0",
        "uv0",
        "texcoord",
        "texcoord_0",
        "texcoord0",
        "map1",
    ):
        candidate = primvars.GetPrimvar(name)
        if (
            candidate
            and candidate.HasValue()
            and str(candidate.GetTypeName()).lower() in texcoord_types
        ):
            st = candidate
            break
    if st is None:
        candidates = [
            candidate
            for candidate in primvars.GetPrimvarsWithValues()
            if str(candidate.GetTypeName()).lower() in texcoord_types
        ]
        candidates.sort(
            key=lambda candidate: (
                not re.match(
                    r"^(st|uv|texcoord)_?\d+$", str(candidate.GetPrimvarName()).lower()
                ),
                str(candidate.GetPrimvarName()).lower(),
            )
        )
        if candidates:
            st = candidates[0]
    texcoords = None
    texcoord_interpolation = ""
    if st:
        texcoord_interpolation = str(st.GetInterpolation())
        texcoords = _expanded_attribute(
            st.Get(),
            st.GetIndices(),
            texcoord_interpolation,
            point_indices,
            face_ids,
            corner_ids,
            2,
        )

    left_handed = str(mesh.GetOrientationAttr().Get()) == "leftHanded"
    holes = set(int(i) for i in (mesh.GetHoleIndicesAttr().Get() or []))
    reverse_winding = left_handed
    if not holes and np.all(counts == 3):
        corners = np.arange(len(point_indices), dtype=np.int64).reshape(-1, 3)
        if reverse_winding:
            corners = corners[:, (0, 2, 1)]
        triangles = np.column_stack((corners, np.arange(len(counts), dtype=np.int64)))
    else:
        triangles: list[tuple[int, int, int, int]] = []
        offset = 0
        for face, count in enumerate(counts):
            count = int(count)
            if face not in holes and count >= 3:
                for i in range(1, count - 1):
                    tri = (
                        (offset, offset + i + 1, offset + i)
                        if reverse_winding
                        else (offset, offset + i, offset + i + 1)
                    )
                    triangles.append((*tri, face))
            offset += count

    # Preserve authored normals regardless of the texture-coordinate
    # interpolation. Face-varying UVs do not imply flat shading, and replacing
    # valid normals with winding-derived ones can invert the shading frame on
    # otherwise valid USD meshes. Generate polygon normals only as a fallback.
    if normals is None:
        normals = np.zeros_like(vertices)
        offset = 0
        for face, count in enumerate(counts):
            count = int(count)
            if face not in holes and count >= 3:
                first, second, third = offset, offset + 1, offset + 2
                face_normal = np.cross(
                    vertices[second] - vertices[first],
                    vertices[third] - vertices[first],
                )
                if left_handed:
                    face_normal = -face_normal
                length = float(np.linalg.norm(face_normal))
                if length > 1.0e-20:
                    normals[offset : offset + count] = face_normal / length
            offset += count
    return vertices, normals, texcoords, triangles, point_indices


def _compact_corners(vertices, normals, texcoords, point_indices, used, corner_remap):
    """Merge corners that have exactly the same position source and attributes."""
    key_parts = [np.asarray(point_indices[used], dtype=np.uint32).reshape(-1, 1)]
    if normals is not None:
        normal_bits = (
            np.ascontiguousarray(normals[used], dtype=np.float32)
            .view(np.uint32)
            .reshape(-1, 3)
        )
        key_parts.append(normal_bits)
    if texcoords is not None:
        texcoord_bits = (
            np.ascontiguousarray(texcoords[used], dtype=np.float32)
            .view(np.uint32)
            .reshape(-1, 2)
        )
        key_parts.append(texcoord_bits)
    keys = np.concatenate(key_parts, axis=1)
    _, first, compact_remap = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    selected = used[first]
    triangles = compact_remap[corner_remap].reshape(-1, 3).astype(np.uint32)
    return selected, triangles


def _normalize_mesh_uvs_for_udim_atlas(
    texcoords: np.ndarray | None, columns: int, rows: int
) -> np.ndarray | None:
    """Map authored UDIM coordinates into the stitched texture atlas."""
    if texcoords is None or len(texcoords) == 0 or (columns <= 1 and rows <= 1):
        return texcoords
    # Use the texture atlas dimensions, not an individual mesh's UV bounds.
    # A mesh contained in tile 1001 still occupies only the first cell of a
    # multi-tile atlas; inferring a divisor of one stretches every tile over
    # that mesh (notably A3's steering wheel and other small parts).
    scale = np.asarray((max(1, columns), max(1, rows)), dtype=np.float32)
    return np.ascontiguousarray(texcoords / scale, dtype=np.float32)


def load_scene_from_usd(
    scene: Scene,
    usd_path: str | Path,
    root_transform: np.ndarray | None = None,
    purposes: tuple[str, ...] = ("default", "render"),
    apply_stage_units: bool = True,
    convert_up_axis: bool = True,
    max_texture_size: int | None = None,
    max_texture_memory_bytes: int | None = None,
    strict_sidedness: bool = False,
    enable_emissive_materials: bool = True,
    load_usd_lights: bool = False,
    usd_light_radius: float = 0.05,
) -> bool:
    """Load composed USD meshes and common PreviewSurface/MDL PBR materials."""
    path = Path(usd_path).expanduser().resolve()
    if not path.is_file():
        return False
    _Sdf, Usd, UsdGeom, UsdLux, UsdShade = _import_usd()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"OpenUSD could not open stage: {path}")
    if max_texture_size is not None and int(max_texture_size) <= 0:
        max_texture_size = None
    if max_texture_memory_bytes is not None and int(max_texture_memory_bytes) <= 0:
        raise ValueError("max_texture_memory_bytes must be positive")
    usd_light_radius = float(usd_light_radius)
    if usd_light_radius <= 0.0:
        raise ValueError("usd_light_radius must be positive")
    scene.usd_environment_path = None
    scene.usd_ambient_light = (0.0, 0.0, 0.0)

    # Omniverse/Isaac stages commonly author a global ambient term in the
    # root layer's renderSettings. It is independent of any DomeLight.
    scene.usd_ambient_light = _ambient_light_from_custom_layer_data(
        stage.GetRootLayer().customLayerData
    )

    start = time.perf_counter()
    # OpenUSD matrices multiply row vectors while the public path-tracer API
    # accepts conventional column-vector transforms, hence the transpose.
    extra_transform = (
        np.eye(4, dtype=np.float64)
        if root_transform is None
        else np.asarray(root_transform, dtype=np.float64).reshape(4, 4).T
    )
    if convert_up_axis:
        up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
        axis_transform = np.eye(4, dtype=np.float64)
        if up_axis == "Z":
            # (x, y, z) -> (x, z, -y)
            axis_transform[:3, :3] = ((1, 0, 0), (0, 0, -1), (0, 1, 0))
        elif up_axis == "X":
            # (x, y, z) -> (-y, x, z)
            axis_transform[:3, :3] = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))
        extra_transform = extra_transform @ axis_transform
    unit_scale = 1.0
    if apply_stage_units:
        unit_scale = float(UsdGeom.GetStageMetersPerUnit(stage))
        extra_transform = extra_transform @ np.diag(
            (unit_scale, unit_scale, unit_scale, 1.0)
        )
    texture_paths: dict[Any, int] = {}
    packed_orm_paths: dict[tuple, int] = {}
    missing_texture_paths: set[Path] = set()
    texture_specs: list[Path | tuple] = []
    srgb_textures: set[int] = set()

    def resolve_texture(texture_path: Path | None):
        if texture_path is None:
            return None
        resolved = _rebase_missing_texture(path, texture_path)
        if "<UDIM>" in str(resolved):
            pattern = resolved.name.replace("<UDIM>", "*")
            expression = re.compile(
                "^"
                + re.escape(resolved.name).replace(re.escape("<UDIM>"), r"(1\d{3})")
                + "$"
            )
            matches = []
            for candidate in resolved.parent.glob(pattern):
                match = expression.match(candidate.name)
                if match:
                    matches.append((int(match.group(1)), candidate.resolve()))
            matches.sort()
            if matches:
                return ("udim", tuple(matches))
        if resolved in missing_texture_paths:
            return None
        if not resolved.is_file():
            logger.warning("USD texture does not exist: %s", resolved)
            missing_texture_paths.add(resolved)
            return None
        return resolved

    def texture_index(texture_path: Path, srgb: bool) -> int:
        resolved = resolve_texture(texture_path)
        if resolved is None:
            return -1
        if resolved not in texture_paths:
            texture_paths[resolved] = len(texture_specs)
            texture_specs.append(resolved)
        index = texture_paths[resolved]
        if srgb:
            srgb_textures.add(index)
        if isinstance(resolved, tuple) and resolved[0] == "udim":
            numbers = [number for number, _ in resolved[1]]
            columns = max((number - 1001) % 10 for number in numbers) + 1
            rows = max((number - 1001) // 10 for number in numbers) + 1
            return index, columns, rows
        return index

    def packed_orm_index(
        roughness_path: Path | None,
        metallic_path: Path | None,
        roughness_constant: float,
        metallic_constant: float,
        roughness_influence: float,
        metallic_influence: float,
    ) -> int:
        roughness_path = resolve_texture(roughness_path)
        metallic_path = resolve_texture(metallic_path)
        if roughness_path is None and metallic_path is None:
            return -1
        key = (
            "packed_orm",
            roughness_path,
            metallic_path,
            float(roughness_constant),
            float(metallic_constant),
            float(roughness_influence),
            float(metallic_influence),
        )
        if key not in packed_orm_paths:
            packed_orm_paths[key] = len(texture_specs)
            texture_specs.append(key)
        return packed_orm_paths[key]

    material_ids: dict[str, int] = {}
    material_uv_grids: dict[int, tuple[int, int]] = {}
    node_paths: list[str] = []
    node_parents: list[int] = []
    node_local_transforms: list[np.ndarray] = []
    node_world_transforms: list[np.ndarray] = []
    path_to_node: dict[str, int] = {}
    instance_ids: list[int] = []
    instance_node_ids: list[int] = []

    def ensure_transform_node(prim) -> int | None:
        """Create a topologically ordered handle for a transformable prim."""
        if not prim or prim.IsPseudoRoot():
            return None
        key = str(prim.GetPath())
        if key in path_to_node:
            return path_to_node[key]
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return ensure_transform_node(prim.GetParent())

        local = (
            np.asarray(xformable.GetLocalTransformation(), dtype=np.float64)
            .reshape(4, 4)
            .T
        )
        parent_index = None
        if not xformable.GetResetXformStack():
            parent_index = ensure_transform_node(prim.GetParent())
        if parent_index is None:
            # Unit/up-axis/root conversion is a stage-root transform in the
            # renderer's conventional column-vector representation.
            local = extra_transform.T @ local
            parent_index = -1
        node_index = len(node_paths)
        path_to_node[key] = node_index
        node_paths.append(key)
        node_parents.append(parent_index)
        local = np.asarray(local, dtype=np.float32)
        node_local_transforms.append(local)
        world = (
            local if parent_index < 0 else node_world_transforms[parent_index] @ local
        )
        node_world_transforms.append(np.asarray(world, dtype=np.float32))
        return node_index

    def material_id(material) -> int:
        if not material:
            key = ""
            if key not in material_ids:
                material_ids[key] = scene.materials.add_default()
            return material_ids[key]
        key = str(material.GetPath())
        if key not in material_ids:
            kwargs = _material_to_pbr(
                material,
                UsdShade,
                texture_index,
                packed_orm_index,
                enable_emissive_materials,
            )
            columns = 1
            rows = 1
            for texture_name in (
                "base_color_texture",
                "normal_texture",
                "metallic_roughness_texture",
                "emissive_texture",
                "occlusion_texture",
            ):
                grid = kwargs.get(texture_name, {}).get("udim_grid")
                if grid:
                    columns = max(columns, int(grid[0]))
                    rows = max(rows, int(grid[1]))
            if columns > 1 or rows > 1:
                for texture_name in (
                    "base_color_texture",
                    "normal_texture",
                    "metallic_roughness_texture",
                    "emissive_texture",
                    "occlusion_texture",
                ):
                    texture_info = kwargs.get(texture_name, {})
                    tile = texture_info.get("udim_tile")
                    if tile is None:
                        continue
                    tile_offset = int(tile) - 1001
                    tile_u = tile_offset % 10
                    tile_v = tile_offset // 10
                    texture_info["transform"] = {
                        "scale": (columns, rows),
                        "offset": (-tile_u, -tile_v),
                    }
            result = scene.materials.add_gltf_material(**kwargs)
            material_ids[key] = result
            material_uv_grids[result] = (columns, rows)
        return material_ids[key]

    stats = {"meshes": 0, "triangles": 0, "vertices": 0, "lights": 0}
    predicate = Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract
    prim_range = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(predicate))
    for prim in prim_range:
        node_index = ensure_transform_node(prim)
        if scene.usd_environment_path is None:
            dome_texture = _dome_texture(prim, UsdLux)
            if dome_texture is not None:
                resolved = (
                    dome_texture
                    if dome_texture.is_absolute()
                    else path.parent / dome_texture
                )
                resolved = resolved.resolve()
                if resolved.is_file():
                    scene.usd_environment_path = resolved
                else:
                    logger.warning("USD DomeLight texture does not exist: %s", resolved)
        if load_usd_lights and prim.IsA(UsdLux.SphereLight):
            imageable = UsdGeom.Imageable(prim)
            if str(imageable.ComputeVisibility()) == "invisible":
                continue
            light = UsdLux.SphereLight(prim)
            color = _as_float_tuple(light.GetColorAttr().Get(), 3, (1.0, 1.0, 1.0))
            if bool(light.GetEnableColorTemperatureAttr().Get()):
                temperature = float(light.GetColorTemperatureAttr().Get() or 6500.0)
                temperature_color = _as_float_tuple(
                    UsdLux.BlackbodyTemperatureAsRgb(temperature),
                    3,
                    (1.0, 1.0, 1.0),
                )
                color = tuple(
                    a * b for a, b in zip(color, temperature_color, strict=True)
                )
            intensity = max(0.0, float(light.GetIntensityAttr().Get() or 0.0))
            intensity *= 2.0 ** float(light.GetExposureAttr().Get() or 0.0)
            intensity = _sphere_light_proxy_radiance(
                intensity,
                bool(light.GetNormalizeAttr().Get()),
                max(0.0, float(light.GetRadiusAttr().Get() or 0.0)),
                node_world_transforms[node_index],
                usd_light_radius,
            )
            position = tuple(
                float(value) for value in node_world_transforms[node_index][:3, 3]
            )
            scene.add_light_sphere(
                position=position,
                radius=usd_light_radius,
                color=color,
                intensity=intensity,
            )
            stats["lights"] += 1
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if (
            str(imageable.ComputeVisibility()) == "invisible"
            or str(imageable.ComputePurpose()) not in purposes
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        node_index = ensure_transform_node(prim)
        arrays = _mesh_arrays(mesh, UsdGeom)
        if arrays is None:
            continue
        vertices, normals, texcoords, triangles, point_indices = arrays

        bound = _bound_material(prim, UsdShade)
        face_materials = np.full(
            len(mesh.GetFaceVertexCountsAttr().Get()),
            material_id(bound),
            dtype=np.int64,
        )
        for subset in _material_bind_subsets(prim, UsdGeom, UsdShade):
            subset_material = _bound_material(subset.GetPrim(), UsdShade)
            if subset_material:
                subset_faces = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
                face_materials[subset_faces] = material_id(subset_material)

        grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for a, b, c, face in triangles:
            grouped[int(face_materials[face])].append((a, b, c))
        for mat_id, group in grouped.items():
            tri = np.asarray(group, dtype=np.int64)
            used, corner_remap = np.unique(tri.reshape(-1), return_inverse=True)
            selected, tri = _compact_corners(
                vertices, normals, texcoords, point_indices, used, corner_remap
            )
            from .scene import Mesh  # noqa: PLC0415

            columns, rows = material_uv_grids.get(mat_id, (1, 1))
            material_texcoords = _normalize_mesh_uvs_for_udim_atlas(
                texcoords, columns, rows
            )
            selected_texcoords = (
                material_texcoords[selected] if material_texcoords is not None else None
            )
            selected_normals = normals[selected] if normals is not None else None
            # A reflected instance reverses geometric winding in world space,
            # while inverse-transpose normal transformation does not include
            # that orientation sign. Apply it to the object-space shading
            # frame so instanced USD meshes match an equivalent baked GLB.
            if (
                selected_normals is not None
                and np.linalg.det(node_world_transforms[node_index][:3, :3]) < 0.0
            ):
                selected_normals = -selected_normals
            out_mesh = Mesh(
                vertices[selected],
                tri,
                normals=selected_normals,
                texcoords=selected_texcoords,
                material_id=mat_id,
            )
            instance_id = scene.add_instance(
                scene.add_mesh(out_mesh),
                # Robust USD rendering is two-sided by default. The closest-
                # hit shader face-forwards the complete shading frame, which
                # tolerates stale orientation/sidedness metadata without a
                # costly mesh-wide consistency scan during import.
                double_sided=(
                    bool(mesh.GetDoubleSidedAttr().Get()) if strict_sidedness else True
                ),
            )
            instance_ids.append(instance_id)
            instance_node_ids.append(node_index)
            stats["meshes"] += 1
            stats["vertices"] += len(selected)
            stats["triangles"] += len(tri)

    from .usd_scene import USDScene  # noqa: PLC0415

    usd_scene = USDScene(scene, stage, str(path), node_paths)
    scene.configure_usd_transform_hierarchy(
        usd_scene,
        np.asarray(node_parents, dtype=np.int32),
        np.asarray(node_local_transforms, dtype=np.float32),
        np.asarray(instance_ids, dtype=np.int32),
        np.asarray(instance_node_ids, dtype=np.int32),
    )

    def decode_texture_spec(spec):
        if isinstance(spec, Path):
            return _decode_texture(spec, max_size=max_texture_size)
        if spec[0] == "udim":
            return _decode_udim(spec[1], max_size=max_texture_size)
        return _decode_packed_orm(spec, max_size=max_texture_size)

    if texture_specs:
        # Decoded 2K RGBA textures and UDIM atlases are large. Keep only two
        # concurrent decodes so production scenes do not multiply peak RAM.
        workers = max(1, min(2, len(texture_specs)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            textures = list(executor.map(decode_texture_spec, texture_specs))
    else:
        textures = []
    textures = _fit_textures_to_budget(
        textures,
        max_bytes=max_texture_memory_bytes,
    )
    scene.set_gltf_textures(textures, srgb_texture_indices=srgb_textures)
    logger.info(
        "[USD timing] total=%.1f ms meshes=%d verts=%d tris=%d lights=%d materials=%d textures=%d",
        (time.perf_counter() - start) * 1000.0,
        stats["meshes"],
        stats["vertices"],
        stats["triangles"],
        stats["lights"],
        len(material_ids),
        len(textures),
    )
    return scene.mesh_count > 0
