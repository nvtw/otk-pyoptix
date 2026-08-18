# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Scene management for path tracing viewer.
Handles mesh loading, BLAS/TLAS construction, and instance management.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import numpy as np
import warp as wp

from .color_utils import srgb_to_linear_rgb
from .materials import MaterialManager

logger = logging.getLogger(__name__)


@wp.kernel
def _update_device_instance_transforms(
    instance_ids: wp.array(dtype=wp.int32),
    xforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    instance_records: wp.array2d(dtype=wp.float32),
    render_nodes: wp.array2d(dtype=wp.float32),
    packed_transforms: wp.array2d(dtype=wp.float32),
    global_rotation: wp.mat33,
    global_translation: wp.vec3,
):
    batch_index = wp.tid()
    instance_index = instance_ids[batch_index]
    xform = xforms[batch_index]
    position = (
        global_rotation * wp.transform_get_translation(xform) + global_translation
    )
    rotation = global_rotation * wp.quat_to_matrix(wp.transform_get_rotation(xform))
    scale = scales[batch_index]

    m00 = rotation[0, 0] * scale[0]
    m01 = rotation[0, 1] * scale[1]
    m02 = rotation[0, 2] * scale[2]
    m10 = rotation[1, 0] * scale[0]
    m11 = rotation[1, 1] * scale[1]
    m12 = rotation[1, 2] * scale[2]
    m20 = rotation[2, 0] * scale[0]
    m21 = rotation[2, 1] * scale[1]
    m22 = rotation[2, 2] * scale[2]

    instance_records[instance_index, 0] = m00
    instance_records[instance_index, 1] = m01
    instance_records[instance_index, 2] = m02
    instance_records[instance_index, 3] = position[0]
    instance_records[instance_index, 4] = m10
    instance_records[instance_index, 5] = m11
    instance_records[instance_index, 6] = m12
    instance_records[instance_index, 7] = position[1]
    instance_records[instance_index, 8] = m20
    instance_records[instance_index, 9] = m21
    instance_records[instance_index, 10] = m22
    instance_records[instance_index, 11] = position[2]

    for column in range(12):
        packed_transforms[instance_index, column] = instance_records[
            instance_index, column
        ]

    render_nodes[instance_index, 0] = m00
    render_nodes[instance_index, 1] = m01
    render_nodes[instance_index, 2] = m02
    render_nodes[instance_index, 3] = position[0]
    render_nodes[instance_index, 4] = m10
    render_nodes[instance_index, 5] = m11
    render_nodes[instance_index, 6] = m12
    render_nodes[instance_index, 7] = position[1]
    render_nodes[instance_index, 8] = m20
    render_nodes[instance_index, 9] = m21
    render_nodes[instance_index, 10] = m22
    render_nodes[instance_index, 11] = position[2]
    render_nodes[instance_index, 12] = 0.0
    render_nodes[instance_index, 13] = 0.0
    render_nodes[instance_index, 14] = 0.0
    render_nodes[instance_index, 15] = 1.0

    inv_sx = 0.0
    inv_sy = 0.0
    inv_sz = 0.0
    if wp.abs(scale[0]) > 1.0e-12:
        inv_sx = 1.0 / scale[0]
    if wp.abs(scale[1]) > 1.0e-12:
        inv_sy = 1.0 / scale[1]
    if wp.abs(scale[2]) > 1.0e-12:
        inv_sz = 1.0 / scale[2]

    i00 = rotation[0, 0] * inv_sx
    i01 = rotation[1, 0] * inv_sx
    i02 = rotation[2, 0] * inv_sx
    i10 = rotation[0, 1] * inv_sy
    i11 = rotation[1, 1] * inv_sy
    i12 = rotation[2, 1] * inv_sy
    i20 = rotation[0, 2] * inv_sz
    i21 = rotation[1, 2] * inv_sz
    i22 = rotation[2, 2] * inv_sz

    render_nodes[instance_index, 16] = i00
    render_nodes[instance_index, 17] = i01
    render_nodes[instance_index, 18] = i02
    render_nodes[instance_index, 19] = -(
        i00 * position[0] + i01 * position[1] + i02 * position[2]
    )
    render_nodes[instance_index, 20] = i10
    render_nodes[instance_index, 21] = i11
    render_nodes[instance_index, 22] = i12
    render_nodes[instance_index, 23] = -(
        i10 * position[0] + i11 * position[1] + i12 * position[2]
    )
    render_nodes[instance_index, 24] = i20
    render_nodes[instance_index, 25] = i21
    render_nodes[instance_index, 26] = i22
    render_nodes[instance_index, 27] = -(
        i20 * position[0] + i21 * position[1] + i22 * position[2]
    )
    render_nodes[instance_index, 28] = 0.0
    render_nodes[instance_index, 29] = 0.0
    render_nodes[instance_index, 30] = 0.0
    render_nodes[instance_index, 31] = 1.0


@wp.kernel
def _scatter_usd_local_transforms(
    transform_count: wp.array(dtype=wp.int32),
    node_ids: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.mat44),
    local_transforms: wp.array(dtype=wp.mat44),
):
    update_index = wp.tid()
    if update_index >= wp.clamp(transform_count[0], 0, node_ids.shape[0]):
        return
    local_transforms[node_ids[update_index]] = transforms[update_index]


@wp.kernel
def _scatter_usd_local_transform_trs(
    transform_count: wp.array(dtype=wp.int32),
    node_ids: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    local_transforms: wp.array(dtype=wp.mat44),
):
    update_index = wp.tid()
    if update_index >= wp.clamp(transform_count[0], 0, node_ids.shape[0]):
        return
    transform = transforms[update_index]
    position = wp.transform_get_translation(transform)
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(transform))
    scale = scales[update_index]
    local_transforms[node_ids[update_index]] = wp.mat44(
        rotation[0, 0] * scale[0],
        rotation[0, 1] * scale[1],
        rotation[0, 2] * scale[2],
        position[0],
        rotation[1, 0] * scale[0],
        rotation[1, 1] * scale[1],
        rotation[1, 2] * scale[2],
        position[1],
        rotation[2, 0] * scale[0],
        rotation[2, 1] * scale[1],
        rotation[2, 2] * scale[2],
        position[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def _compose_usd_transform_level(
    level_nodes: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    local_transforms: wp.array(dtype=wp.mat44),
    world_transforms: wp.array(dtype=wp.mat44),
):
    node_index = level_nodes[wp.tid()]
    parent_index = parents[node_index]
    if parent_index < 0:
        world_transforms[node_index] = local_transforms[node_index]
    else:
        world_transforms[node_index] = (
            world_transforms[parent_index] * local_transforms[node_index]
        )


@wp.kernel
def _write_usd_instance_transforms(
    instance_ids: wp.array(dtype=wp.int32),
    instance_node_ids: wp.array(dtype=wp.int32),
    world_transforms: wp.array(dtype=wp.mat44),
    instance_records: wp.array2d(dtype=wp.float32),
    render_nodes: wp.array2d(dtype=wp.float32),
    packed_transforms: wp.array2d(dtype=wp.float32),
):
    mapping_index = wp.tid()
    instance_index = instance_ids[mapping_index]
    matrix = world_transforms[instance_node_ids[mapping_index]]
    inverse = wp.inverse(matrix)

    for row in range(3):
        for column in range(4):
            value = matrix[row, column]
            flat_index = row * 4 + column
            instance_records[instance_index, flat_index] = value
            packed_transforms[instance_index, flat_index] = value

    for row in range(4):
        for column in range(4):
            flat_index = row * 4 + column
            render_nodes[instance_index, flat_index] = matrix[row, column]
            render_nodes[instance_index, 16 + flat_index] = inverse[row, column]


@wp.kernel
def _update_device_instance_visibility(
    instance_ids: wp.array(dtype=wp.int32),
    visibility_mask: wp.uint32,
    instance_record_words: wp.array2d(dtype=wp.uint32),
):
    instance_index = instance_ids[wp.tid()]
    instance_record_words[instance_index, 14] = visibility_mask


@wp.func
def _srgb_channel_to_linear(value: float):
    if value <= 0.04045:
        return value / 12.92
    return wp.pow((value + 0.055) / 1.055, 2.4)


@wp.kernel
def _update_device_instance_materials(
    material_ids: wp.array(dtype=wp.int32),
    colors: wp.array(dtype=wp.vec3),
    properties: wp.array(dtype=wp.vec4),
    color_count: int,
    property_count: int,
    compact_materials: wp.array2d(dtype=wp.float32),
):
    thread_index = wp.tid()
    color_index = 0 if color_count == 1 else thread_index
    property_index = 0 if property_count == 1 else thread_index
    color = colors[color_index]
    material = properties[property_index]
    material_index = material_ids[thread_index]
    compact_materials[material_index, 0] = _srgb_channel_to_linear(color[0])
    compact_materials[material_index, 1] = _srgb_channel_to_linear(color[1])
    compact_materials[material_index, 2] = _srgb_channel_to_linear(color[2])
    compact_materials[material_index, 6] = wp.clamp(material[0], 0.0, 1.0)
    compact_materials[material_index, 7] = wp.clamp(material[1], 0.0, 1.0)
    compact_materials[material_index, 8] = wp.max(material[2], 0.0)
    compact_materials[material_index, 9] = wp.max(material[3], 0.0)


def _create_vertex_buffers_dtype():
    """Create numpy dtype for VertexBuffers structure (offset-based)."""
    return np.dtype(
        [
            ("positionOffset", np.uint32),
            ("normalOffset", np.uint32),
            ("colorOffset", np.uint32),
            ("tangentOffset", np.uint32),
            ("texCoord0Offset", np.uint32),
            ("texCoord1Offset", np.uint32),
            ("prevPositionOffset", np.uint32),
            ("hasTexCoord1", np.uint32),
            ("hasPrevPosition", np.uint32),
        ]
    )


def _create_render_primitive_dtype():
    """Create numpy dtype for RenderPrimitive structure."""
    vb_dt = _create_vertex_buffers_dtype()
    return np.dtype(
        [
            ("indexOffset", np.uint32),
            ("materialIdOffset", np.uint32),
            ("vertexBuffer", vb_dt),
            ("numIndices", np.uint32),
            ("numVertices", np.uint32),
        ]
    )


def _create_render_node_dtype():
    """Create numpy dtype for RenderNode structure (136 bytes)."""
    return np.dtype(
        [
            ("objectToWorld", np.float32, (4, 4)),
            ("worldToObject", np.float32, (4, 4)),
            ("materialID", np.int32),
            ("renderPrimID", np.int32),
        ]
    )


def _create_scene_description_dtype():
    """Create numpy dtype for SceneDescription structure (48 bytes)."""
    return np.dtype(
        [
            ("materialAddress", np.uint64),
            ("renderNodeAddress", np.uint64),
            ("renderPrimitiveAddress", np.uint64),
            ("lightAddress", np.uint64),
            ("numLights", np.int32),
            ("_padding", np.int32),
        ]
    )


def _build_optix_instance_dtype() -> np.dtype:
    """ABI-compatible OptiX instance layout (80 bytes)."""
    names = [
        "transform",
        "instanceId",
        "sbtOffset",
        "visibilityMask",
        "flags",
        "traversableHandle",
    ]
    formats = [("f4", (12,)), "u4", "u4", "u4", "u4", "u8"]
    offsets = [0, 48, 52, 56, 60, 64]
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": 80}
    )


class Mesh:
    """Represents a mesh with vertices, indices, and GPU buffers."""

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray = None,
        texcoords: np.ndarray = None,
        texcoords1: np.ndarray = None,
        material_id: int = 0,
    ):
        """
        Create a mesh.

        Args:
            vertices: Nx3 array of vertex positions
            indices: Mx3 array of triangle indices
            normals: Nx3 array of vertex normals (optional, will compute if None)
            texcoords: Nx2 array of texture coordinates (optional)
            material_id: Material index for this mesh
        """
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32)

        if normals is None:
            normals = self._compute_normals(self.vertices, self.indices)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)

        if texcoords is None:
            texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords = np.ascontiguousarray(texcoords, dtype=np.float32)
        if texcoords1 is None:
            texcoords1 = np.zeros((len(vertices), 2), dtype=np.float32)
        self.texcoords1 = np.ascontiguousarray(texcoords1, dtype=np.float32)

        self.material_id = material_id

        # Tangents for normal mapping (computed from UVs; used when normalTexIndex >= 0)
        self.tangents = self._compute_tangents(
            self.vertices, self.indices, self.normals, self.texcoords
        )

        # GPU buffers (created on build)
        self.d_vertices = None
        self.d_indices = None
        self.d_normals = None
        self.d_tangents = None
        self.d_texcoords = None
        self.d_texcoords1 = None
        self.d_material_ids = None

    def _compute_tangents(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray,
        texcoords: np.ndarray,
    ) -> np.ndarray:
        """Compute vertex tangents from UVs using vectorized operations."""
        n_verts = len(vertices)
        tangents = np.zeros((n_verts, 4), dtype=np.float32)
        tan1 = np.zeros((n_verts, 3), dtype=np.float32)
        tan2 = np.zeros((n_verts, 3), dtype=np.float32)

        tri = indices.astype(np.int64, copy=False)
        i0 = tri[:, 0]
        i1 = tri[:, 1]
        i2 = tri[:, 2]

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        uv0 = texcoords[i0]
        uv1 = texcoords[i1]
        uv2 = texcoords[i2]

        e1 = v1 - v0
        e2 = v2 - v0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0

        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        valid = np.abs(denom) >= 1.0e-8
        if np.any(valid):
            inv = np.zeros_like(denom, dtype=np.float32)
            inv[valid] = 1.0 / denom[valid]
            inv3 = inv[:, None]
            tri_tangent = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) * inv3
            tri_bitangent = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) * inv3
            tri_tangent[~valid] = 0.0
            tri_bitangent[~valid] = 0.0

            np.add.at(tan1, i0, tri_tangent)
            np.add.at(tan1, i1, tri_tangent)
            np.add.at(tan1, i2, tri_tangent)
            np.add.at(tan2, i0, tri_bitangent)
            np.add.at(tan2, i1, tri_bitangent)
            np.add.at(tan2, i2, tri_bitangent)

        n = normals
        tangent = tan1 - n * np.sum(n * tan1, axis=1, keepdims=True)
        t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
        good = t_len[:, 0] > 1.0e-8
        tangent[good] /= t_len[good]

        fallback_x = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_verts, 1))
        fallback_y = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_verts, 1))
        use_y = np.abs(n[:, 0]) > 0.9
        tangent[~good] = np.where(
            use_y[~good, None], fallback_y[~good], fallback_x[~good]
        )

        cross_nt = np.cross(n, tangent)
        handedness = np.where(np.sum(cross_nt * tan2, axis=1) < 0.0, -1.0, 1.0).astype(
            np.float32
        )

        tangents[:, :3] = tangent
        tangents[:, 3] = handedness
        return np.ascontiguousarray(tangents, dtype=np.float32)

    def _compute_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute vertex normals from face normals."""
        normals = np.zeros_like(vertices)

        for tri in indices:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            face_normal = np.cross(e1, e2)

            normals[tri[0]] += face_normal
            normals[tri[1]] += face_normal
            normals[tri[2]] += face_normal

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals = normals / lengths

        return normals

    def upload_to_gpu(self):
        """Upload mesh data to GPU."""
        self.d_vertices = wp.array(
            self.vertices.flatten(), dtype=wp.float32, device="cuda"
        )
        self.d_indices = wp.array(
            self.indices.flatten(), dtype=wp.uint32, device="cuda"
        )
        self.d_normals = wp.array(
            self.normals.flatten(), dtype=wp.float32, device="cuda"
        )
        self.d_tangents = wp.array(
            self.tangents.flatten(), dtype=wp.float32, device="cuda"
        )
        self.d_texcoords = wp.array(
            self.texcoords.flatten(), dtype=wp.float32, device="cuda"
        )
        self.d_texcoords1 = wp.array(
            self.texcoords1.flatten(), dtype=wp.float32, device="cuda"
        )

        # Per-triangle material IDs
        num_triangles = len(self.indices)
        material_ids = np.full(num_triangles, self.material_id, dtype=np.uint32)
        self.d_material_ids = wp.array(material_ids, dtype=wp.uint32, device="cuda")


class Instance:
    """Represents an instance of a mesh with a transform."""

    def __init__(
        self,
        mesh_index: int,
        transform: np.ndarray = None,
        material_id: int | None = None,
        visible: bool = True,
        double_sided: bool = False,
    ):
        """
        Create an instance.

        Args:
            mesh_index: Index of the mesh in the scene
            transform: 4x4 transformation matrix (identity if None)
        """
        self.mesh_index = mesh_index
        self.material_id = material_id
        self.visible = bool(visible)
        self.double_sided = bool(double_sided)
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        self.transform = np.ascontiguousarray(transform, dtype=np.float32)
        self.prev_transform = self.transform.copy()


class Scene:
    """
    Scene management for path tracing.
    Handles meshes, instances, materials, and acceleration structures.
    """

    def __init__(self, optix_ctx):
        """
        Create a scene.

        Args:
            optix_ctx: OptiX device context
        """
        self._optix = None  # Will be set when building
        self._ctx = optix_ctx
        self.materials = MaterialManager()
        self._meshes = []
        self._instances = []
        self._sphere_lights = []
        self._sphere_light_data = None

        # GPU buffers
        self._render_primitives = None
        self._render_nodes = None
        self._scene_desc = None
        self._instance_material_ids = None
        self._compact_materials = None
        self._compact_material_floats = None
        self._instance_render_prim_ids = None
        self._texture_data = None
        self._texture_objects = []
        self._packed_indices = None
        self._packed_normals = None
        self._packed_tangents = None
        self._packed_texcoords0 = None
        self._packed_texcoords1 = None
        self._packed_prev_positions = None
        self._packed_material_ids = None
        self._gltf_textures = []
        # Resolved DomeLight texture discovered while composing the last USD
        # stage. Keeping it here lets PathTracerAPI opt into the environment
        # without reopening (and recomposing) a large stage.
        self.usd_environment_path = None
        self.usd_ambient_light = (0.0, 0.0, 0.0)
        self.usd_scene = None
        self._usd_node_parents = np.empty(0, dtype=np.int32)
        self._usd_local_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_world_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_instance_ids = np.empty(0, dtype=np.int32)
        self._usd_instance_node_ids = np.empty(0, dtype=np.int32)
        self._usd_level_nodes = []
        self._d_usd_node_parents = None
        self._d_usd_local_transforms = None
        self._d_usd_world_transforms = None
        self._d_usd_instance_ids = None
        self._d_usd_instance_node_ids = None
        self._d_usd_level_nodes = []

        # Acceleration structures
        self._gas_handles = []
        self._gas_buffers = []
        self._ias_handle = None
        self._ias_buffer = None
        self._instance_buffer = None
        self._instance_record_floats = None
        self._instance_record_words = None
        self._render_node_floats = None
        self._device_instance_transforms = None
        self._instance_records_dirty = True
        self._instance_np_cache = None
        self._instance_np_capacity = 0
        self._instance_transform_cache = np.empty((0, 4, 4), dtype=np.float32)
        self._instance_visibility_cache = np.empty(0, dtype=np.bool_)
        self._instance_cache_capacity = 0
        self._tlas_temp_buffer = None
        self._tlas_temp_capacity = 0
        self._tlas_output_capacity = 0
        self._tlas_output_size = 0
        self._tlas_instance_count = 0

        # Keepalive references
        self._keepalive = {}

    @property
    def mesh_count(self) -> int:
        return len(self._meshes)

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    @property
    def light_count(self) -> int:
        """Return the number of analytic lights."""
        return len(self._sphere_lights)

    def add_light_sphere(
        self,
        position: tuple[float, float, float],
        radius: float,
        color: tuple[float, float, float],
        intensity: float,
    ) -> int:
        """Add a constant-radiance analytic sphere light."""
        radius = float(radius)
        intensity = float(intensity)
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if intensity < 0.0:
            raise ValueError("intensity must be nonnegative")
        self._sphere_lights.append(
            (
                tuple(float(value) for value in position),
                radius,
                tuple(float(value) for value in color),
                intensity,
            )
        )
        return len(self._sphere_lights) - 1

    @property
    def tlas_handle(self) -> int:
        """Get the TLAS traversable handle."""
        return self._ias_handle if self._ias_handle else 0

    @property
    def scene_desc_address(self) -> int:
        """Get the device address of the scene description."""
        return self._scene_desc.ptr if self._scene_desc else 0

    @property
    def instance_material_ids_address(self) -> int:
        """Get device address of per-instance material id buffer."""
        return (
            self._instance_material_ids.ptr
            if self._instance_material_ids is not None
            else 0
        )

    @property
    def compact_materials_address(self) -> int:
        """Get device address of compact material table."""
        return self._compact_materials.ptr if self._compact_materials is not None else 0

    @property
    def render_nodes_address(self) -> int:
        """Get device address of render node array."""
        return self._render_nodes.ptr if self._render_nodes is not None else 0

    @property
    def render_primitives_address(self) -> int:
        """Get device address of render primitive array."""
        return self._render_primitives.ptr if self._render_primitives is not None else 0

    @property
    def instance_render_prim_ids_address(self) -> int:
        """Get device address of per-instance render primitive id buffer."""
        return (
            self._instance_render_prim_ids.ptr
            if self._instance_render_prim_ids is not None
            else 0
        )

    @property
    def texture_data_address(self) -> int:
        """Get the device address of the native texture-handle array."""
        return self._texture_data.ptr if self._texture_data is not None else 0

    @property
    def texture_count(self) -> int:
        """Get number of loaded glTF textures."""
        return len(self._gltf_textures)

    @property
    def has_meshes(self) -> bool:
        """Return True when at least one mesh is present."""
        return bool(self._meshes)

    def get_instance_material_ids_host(self) -> np.ndarray | None:
        """Return per-instance material IDs as a host NumPy array copy."""
        if self._instance_material_ids is None:
            return None
        return self._instance_material_ids.numpy()

    def set_instance_material_ids_host(self, material_ids: np.ndarray):
        """Upload per-instance material IDs from host memory."""
        self._instance_material_ids = wp.array(
            np.asarray(material_ids, dtype=np.uint32), dtype=wp.uint32, device="cuda"
        )

    def set_compact_material_bytes(self, compact_bytes: np.ndarray):
        """Upload compact material table bytes to GPU."""
        self._compact_materials = wp.array(
            np.asarray(compact_bytes, dtype=np.uint8), dtype=wp.uint8, device="cuda"
        )
        self._compact_material_floats = None

    def set_instance_materials_device(
        self,
        material_ids: wp.array,
        colors: wp.array,
        properties: wp.array,
    ) -> bool:
        """Update instance-owned compact material fields directly on CUDA."""
        if len(material_ids) == 0:
            return True
        if self._compact_materials is None:
            return False
        material_count = self.materials.count
        stride = int(self._compact_materials.capacity) // material_count // 4
        self._compact_material_floats = wp.array(
            ptr=self._compact_materials.ptr,
            shape=(material_count, stride),
            dtype=wp.float32,
            capacity=int(self._compact_materials.capacity),
            device="cuda",
        )
        if len(colors) not in (1, len(material_ids)):
            raise ValueError("Device colors must contain one or one value per instance")
        if len(properties) not in (1, len(material_ids)):
            raise ValueError(
                "Device material properties must contain one or one value per instance"
            )
        wp.launch(
            _update_device_instance_materials,
            dim=len(material_ids),
            inputs=[
                material_ids,
                colors,
                properties,
                len(colors),
                len(properties),
                self._compact_material_floats,
            ],
            device="cuda",
        )
        return True

    def set_gltf_textures(
        self,
        textures: list[np.ndarray],
        srgb_texture_indices: set[int] | None = None,
        append: bool = False,
    ):
        """
        Set glTF texture list as normalized RGBA8 images.

        Args:
            textures: RGBA uint8 textures, or float textures in normalized [0,1] space.
            srgb_texture_indices: Texture indices that represent color data and
                therefore require sRGB->linear decode to match Vulkan hardware
                sampling with ``R8G8B8A8_SRGB``.
        """
        srgb_indices = srgb_texture_indices or set()
        converted = []
        for tex_idx, tex in enumerate(textures):
            if np.issubdtype(tex.dtype, np.integer):
                tex_u8 = np.ascontiguousarray(tex, dtype=np.uint8)
            else:
                tex_u8 = np.clip(np.asarray(tex) * 255.0 + 0.5, 0.0, 255.0).astype(
                    np.uint8
                )
            if tex_idx in srgb_indices:
                tex_u8 = tex_u8.copy()
                linear = srgb_to_linear_rgb(
                    tex_u8[..., :3].astype(np.float32) * (1.0 / 255.0)
                )
                tex_u8[..., :3] = np.clip(linear * 255.0 + 0.5, 0.0, 255.0).astype(
                    np.uint8
                )
            converted.append(tex_u8)
        if append:
            self._gltf_textures.extend(converted)
        else:
            self._gltf_textures = converted

    def add_mesh(self, mesh: Mesh) -> int:
        """Add a mesh to the scene and return its index."""
        self._meshes.append(mesh)
        return len(self._meshes) - 1

    def _ensure_instance_cache_capacity(self, count: int):
        """Grow contiguous host-side instance state without per-frame allocation."""
        if self._instance_cache_capacity >= count:
            return
        capacity = max(count, 1, self._instance_cache_capacity * 2)
        transforms = np.empty((capacity, 4, 4), dtype=np.float32)
        visibility = np.empty(capacity, dtype=np.bool_)
        if self._instance_cache_capacity:
            transforms[: self._instance_cache_capacity] = self._instance_transform_cache
            visibility[: self._instance_cache_capacity] = (
                self._instance_visibility_cache
            )
        self._instance_transform_cache = transforms
        self._instance_visibility_cache = visibility
        self._instance_cache_capacity = capacity

    def add_instance(
        self,
        mesh_index: int,
        transform: np.ndarray = None,
        material_id: int | None = None,
        visible: bool = True,
        double_sided: bool = False,
    ) -> int:
        """Add an instance of a mesh and return its index."""
        instance = Instance(
            mesh_index,
            transform,
            material_id=material_id,
            visible=visible,
            double_sided=double_sided,
        )
        self._instances.append(instance)
        instance_index = len(self._instances) - 1
        self._ensure_instance_cache_capacity(instance_index + 1)
        self._instance_transform_cache[instance_index] = instance.transform
        self._instance_visibility_cache[instance_index] = instance.visible
        self._instance_records_dirty = True
        return instance_index

    def set_instance_transform(self, instance_index: int, transform: np.ndarray):
        """Update an instance's transform."""
        if 0 <= instance_index < len(self._instances):
            inst = self._instances[instance_index]
            inst.prev_transform = inst.transform.copy()
            inst.transform = np.ascontiguousarray(transform, dtype=np.float32)
            self._instance_transform_cache[instance_index] = inst.transform
            self._instance_records_dirty = True

    def set_instance_transforms_batch(self, instance_indices, transforms: np.ndarray):
        """Update a batch of instance transforms with one NumPy assignment."""
        indices = np.asarray(instance_indices, dtype=np.intp)
        matrices = np.asarray(transforms, dtype=np.float32).reshape(-1, 4, 4)
        if len(indices) != len(matrices):
            raise ValueError("instance_indices and transforms must have equal length")
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._instances):
            raise IndexError("instance index is out of range")
        self._instance_transform_cache[indices] = matrices
        self._instance_records_dirty = True

    def set_instance_transforms_device(
        self,
        instance_indices: wp.array,
        xforms: wp.array,
        scales: wp.array,
        global_transform: np.ndarray,
    ) -> bool:
        """Update traversal, shading, and motion transforms directly on CUDA."""
        if len(instance_indices) == 0:
            return True
        if (
            self._instance_record_floats is None
            or self._render_node_floats is None
            or self._device_instance_transforms is None
        ):
            return False
        if (
            not instance_indices.device.is_cuda
            or not xforms.device.is_cuda
            or not scales.device.is_cuda
        ):
            raise ValueError("Device instance updates require CUDA arrays")
        matrix = np.asarray(global_transform, dtype=np.float32).reshape(4, 4)
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        wp.launch(
            _update_device_instance_transforms,
            dim=len(instance_indices),
            inputs=[
                instance_indices,
                xforms,
                scales,
                self._instance_record_floats,
                self._render_node_floats,
                self._device_instance_transforms,
                wp.mat33(*rotation.reshape(-1)),
                wp.vec3(*translation),
            ],
            device="cuda",
        )
        return True

    def configure_usd_transform_hierarchy(
        self,
        usd_scene,
        parents: np.ndarray,
        local_transforms: np.ndarray,
        instance_ids: np.ndarray,
        instance_node_ids: np.ndarray,
    ):
        """Retain a composed USD transform hierarchy for dynamic updates."""
        parents = np.asarray(parents, dtype=np.int32).reshape(-1)
        local = np.ascontiguousarray(local_transforms, dtype=np.float32).reshape(
            -1, 4, 4
        )
        if len(parents) != len(local):
            raise ValueError(
                "USD hierarchy parents and transforms must have equal length"
            )
        if np.any(parents >= np.arange(len(parents), dtype=np.int32)):
            raise ValueError("USD hierarchy parents must precede their children")

        world = np.empty_like(local)
        depths = np.zeros(len(parents), dtype=np.int32)
        for node_index, parent_index in enumerate(parents):
            if parent_index < 0:
                world[node_index] = local[node_index]
            else:
                world[node_index] = world[parent_index] @ local[node_index]
                depths[node_index] = depths[parent_index] + 1

        self.usd_scene = usd_scene
        self._usd_node_parents = parents
        self._usd_local_transforms = local
        self._usd_world_transforms = world
        self._usd_instance_ids = np.asarray(instance_ids, dtype=np.int32).reshape(-1)
        self._usd_instance_node_ids = np.asarray(
            instance_node_ids, dtype=np.int32
        ).reshape(-1)
        self._usd_level_nodes = [
            np.flatnonzero(depths == depth).astype(np.int32)
            for depth in range(int(depths.max(initial=0)) + 1)
        ]
        if len(self._usd_instance_ids) != len(self._usd_instance_node_ids):
            raise ValueError("USD instance and node mappings must have equal length")
        if len(self._usd_instance_ids):
            self.set_instance_transforms_batch(
                self._usd_instance_ids,
                world[self._usd_instance_node_ids],
            )

    def _build_usd_transform_buffers(self):
        if self.usd_scene is None or len(self._usd_node_parents) == 0:
            return
        self._d_usd_node_parents = wp.array(
            self._usd_node_parents, dtype=wp.int32, device="cuda"
        )
        self._d_usd_local_transforms = wp.array(
            self._usd_local_transforms, dtype=wp.mat44, device="cuda"
        )
        self._d_usd_world_transforms = wp.array(
            self._usd_world_transforms, dtype=wp.mat44, device="cuda"
        )
        self._d_usd_instance_ids = wp.array(
            self._usd_instance_ids, dtype=wp.int32, device="cuda"
        )
        self._d_usd_instance_node_ids = wp.array(
            self._usd_instance_node_ids, dtype=wp.int32, device="cuda"
        )
        self._d_usd_level_nodes = [
            wp.array(level, dtype=wp.int32, device="cuda")
            for level in self._usd_level_nodes
        ]
        self.usd_scene._attach_device_arrays(
            self._d_usd_local_transforms, self._d_usd_world_transforms
        )

    def _evaluate_usd_transform_hierarchy_host(self, update_instances: bool = True):
        for node_index, parent_index in enumerate(self._usd_node_parents):
            local = self._usd_local_transforms[node_index]
            if parent_index < 0:
                self._usd_world_transforms[node_index] = local
            else:
                self._usd_world_transforms[node_index] = (
                    self._usd_world_transforms[parent_index] @ local
                )
        if update_instances and len(self._usd_instance_ids):
            self.set_instance_transforms_batch(
                self._usd_instance_ids,
                self._usd_world_transforms[self._usd_instance_node_ids],
            )

    def _evaluate_usd_transform_hierarchy(self, stream=None):
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        scope = (
            wp.ScopedStream(stream, sync_enter=False, sync_exit=False)
            if stream is not None
            else nullcontext()
        )
        with scope:
            for level_nodes in self._d_usd_level_nodes:
                wp.launch(
                    _compose_usd_transform_level,
                    dim=len(level_nodes),
                    inputs=[
                        level_nodes,
                        self._d_usd_node_parents,
                        self._d_usd_local_transforms,
                        self._d_usd_world_transforms,
                    ],
                    device="cuda",
                )
            if len(self._usd_instance_ids):
                wp.launch(
                    _write_usd_instance_transforms,
                    dim=len(self._usd_instance_ids),
                    inputs=[
                        self._d_usd_instance_ids,
                        self._d_usd_instance_node_ids,
                        self._d_usd_world_transforms,
                        self._instance_record_floats,
                        self._render_node_floats,
                        self._device_instance_transforms,
                    ],
                    device="cuda",
                )

    def set_usd_local_transforms_device(
        self,
        transform_count: wp.array,
        node_ids: wp.array,
        transforms: wp.array,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply CUDA-resident USD local matrices and compose the hierarchy."""
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        if (
            not transform_count.device.is_cuda
            or not node_ids.device.is_cuda
            or not transforms.device.is_cuda
        ):
            raise ValueError("USD device transform updates require CUDA arrays")
        if len(transform_count) != 1:
            raise ValueError("transform_count must contain exactly one value")
        if len(node_ids) != len(transforms):
            raise ValueError("node_ids and transforms must have equal length")
        if len(node_ids) == 0:
            return
        if (
            transform_count.dtype != wp.int32
            or node_ids.dtype != wp.int32
            or transforms.dtype != wp.mat44
        ):
            raise TypeError(
                "USD device batches require an int32 count, int32 IDs, and mat44 matrices"
            )
        scope = (
            wp.ScopedStream(stream, sync_enter=False, sync_exit=False)
            if stream is not None
            else nullcontext()
        )
        with scope:
            wp.launch(
                _scatter_usd_local_transforms,
                dim=len(node_ids),
                inputs=[
                    transform_count,
                    node_ids,
                    transforms,
                    self._d_usd_local_transforms,
                ],
                device="cuda",
            )
            self._evaluate_usd_transform_hierarchy()
            if rebuild_tlas:
                self.rebuild_tlas()

    def set_usd_local_transform_trs_device(
        self,
        transform_count: wp.array,
        node_ids: wp.array,
        transforms: wp.array,
        scales: wp.array,
        stream=None,
        rebuild_tlas: bool = True,
    ):
        """Apply CUDA-resident Warp transforms/scales to USD local nodes."""
        if self._d_usd_local_transforms is None:
            raise RuntimeError("Build the scene before updating USD transforms")
        if (
            not transform_count.device.is_cuda
            or not node_ids.device.is_cuda
            or not transforms.device.is_cuda
            or not scales.device.is_cuda
        ):
            raise ValueError("USD device transform updates require CUDA arrays")
        if len(transform_count) != 1:
            raise ValueError("transform_count must contain exactly one value")
        if len(node_ids) != len(transforms) or len(node_ids) != len(scales):
            raise ValueError("node_ids, transforms, and scales must have equal length")
        if len(node_ids) == 0:
            return
        if (
            transform_count.dtype != wp.int32
            or node_ids.dtype != wp.int32
            or transforms.dtype != wp.transform
            or scales.dtype != wp.vec3
        ):
            raise TypeError(
                "USD TRS batches require an int32 count, int32 IDs, transform poses, and vec3 scales"
            )
        scope = (
            wp.ScopedStream(stream, sync_enter=False, sync_exit=False)
            if stream is not None
            else nullcontext()
        )
        with scope:
            wp.launch(
                _scatter_usd_local_transform_trs,
                dim=len(node_ids),
                inputs=[
                    transform_count,
                    node_ids,
                    transforms,
                    scales,
                    self._d_usd_local_transforms,
                ],
                device="cuda",
            )
            self._evaluate_usd_transform_hierarchy()
            if rebuild_tlas:
                self.rebuild_tlas()

    def set_usd_local_transforms(
        self, node_ids, transforms: np.ndarray, stream=None, rebuild_tlas: bool = True
    ):
        """Upload a host batch and apply it through the CUDA hierarchy path."""
        indices = np.asarray(node_ids, dtype=np.int32).reshape(-1)
        matrices = np.ascontiguousarray(transforms, dtype=np.float32).reshape(-1, 4, 4)
        if len(indices) != len(matrices):
            raise ValueError("node_ids and transforms must have equal length")
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._usd_node_parents):
            raise IndexError("USD transform handle is out of range")
        self._usd_local_transforms[indices] = matrices
        self._evaluate_usd_transform_hierarchy_host(
            update_instances=self._d_usd_local_transforms is None
        )
        if self._d_usd_local_transforms is None:
            return
        scope = (
            wp.ScopedStream(stream, sync_enter=False, sync_exit=False)
            if stream is not None
            else nullcontext()
        )
        with scope:
            device_count = wp.array([len(indices)], dtype=wp.int32, device="cuda")
            device_ids = wp.array(indices, dtype=wp.int32, device="cuda")
            device_matrices = wp.array(matrices, dtype=wp.mat44, device="cuda")
            self.set_usd_local_transforms_device(
                device_count,
                device_ids,
                device_matrices,
                rebuild_tlas=rebuild_tlas,
            )

    def set_instance_material(self, instance_index: int, material_id: int | None):
        """Override the material used by one instance."""
        if 0 <= instance_index < len(self._instances):
            self._instances[instance_index].material_id = material_id

    def set_instance_visible(self, instance_index: int, visible: bool):
        """Control whether one instance participates in traversal."""
        if 0 <= instance_index < len(self._instances):
            value = bool(visible)
            self._instances[instance_index].visible = value
            self._instance_visibility_cache[instance_index] = value
            self._instance_records_dirty = True

    def set_instances_visible_batch(self, instance_indices, visible: bool):
        """Set visibility for a batch with one host-array assignment."""
        indices = np.asarray(instance_indices, dtype=np.intp)
        if len(indices) == 0:
            return
        if int(indices.min()) < 0 or int(indices.max()) >= len(self._instances):
            raise IndexError("instance index is out of range")
        self._instance_visibility_cache[indices] = bool(visible)
        if self._instance_record_words is None:
            self._instance_records_dirty = True
            return
        device_indices = wp.array(
            indices.astype(np.int32), dtype=wp.int32, device="cuda"
        )
        wp.launch(
            _update_device_instance_visibility,
            dim=len(indices),
            inputs=[
                device_indices,
                wp.uint32(0xFF if visible else 0),
                self._instance_record_words,
            ],
            device="cuda",
        )

    def create_cornell_box(self):
        """Create a Cornell Box scene."""
        self.clear()

        # Materials
        white = self.materials.add_diffuse((0.8, 0.8, 0.8))
        red = self.materials.add_diffuse((0.8, 0.1, 0.1))
        green = self.materials.add_diffuse((0.1, 0.8, 0.1))
        light = self.materials.add_emissive((1.0, 0.95, 0.85), 15.0)

        # Floor
        self.add_box((-2, -2, -2), (2, -1.9, 2), white)
        # Ceiling
        self.add_box((-2, 1.9, -2), (2, 2, 2), white)
        # Back wall
        self.add_box((-2, -2, -2), (2, 2, -1.9), white)
        # Left wall (red)
        self.add_box((-2, -2, -2), (-1.9, 2, 2), red)
        # Right wall (green)
        self.add_box((1.9, -2, -2), (2, 2, 2), green)
        # Short box
        self.add_box((-0.8, -1.9, -0.8), (0.2, -0.5, 0.2), white)
        # Tall box
        self.add_box((0.3, -1.9, -1.2), (1.2, 0.3, -0.3), white)
        # Light
        self.add_box((-0.5, 1.85, -0.5), (0.5, 1.89, 0.5), light)

    def load_from_gltf(
        self,
        gltf_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
    ) -> bool:
        """Load a glTF/GLB scene into this scene."""
        from .asset_loaders import load_scene_from_gltf

        if clear_existing:
            self.clear()
        return bool(
            load_scene_from_gltf(self, gltf_path, root_transform=root_transform)
        )

    def load_from_usd(
        self,
        usd_path: str,
        root_transform: np.ndarray | None = None,
        clear_existing: bool = True,
        apply_stage_units: bool = True,
        convert_up_axis: bool = True,
        max_texture_size: int | None = None,
        max_texture_memory_bytes: int | None = None,
        strict_sidedness: bool = False,
        enable_emissive_materials: bool = True,
        load_usd_lights: bool = False,
        usd_light_radius: float = 0.05,
    ) -> bool:
        """Load a composed USD stage into this scene.

        OpenUSD is an optional dependency and is imported only when this method
        is called. Meshes, UVs, material subsets, UsdPreviewSurface materials,
        and common NVIDIA MDL PBR inputs are translated to the internal scene.
        """
        from .usd_loader import load_scene_from_usd

        if clear_existing:
            self.clear()
        return bool(
            load_scene_from_usd(
                self,
                usd_path,
                root_transform=root_transform,
                apply_stage_units=apply_stage_units,
                convert_up_axis=convert_up_axis,
                max_texture_size=max_texture_size,
                max_texture_memory_bytes=max_texture_memory_bytes,
                strict_sidedness=strict_sidedness,
                enable_emissive_materials=enable_emissive_materials,
                load_usd_lights=load_usd_lights,
                usd_light_radius=usd_light_radius,
            )
        )

    def load_from_obj(self, obj_path: str) -> bool:
        """Load an OBJ scene into this scene."""
        from .asset_loaders import load_scene_from_obj

        self.clear()
        return bool(load_scene_from_obj(self, obj_path))

    def add_box(self, min_pt: tuple, max_pt: tuple, material_id: int) -> int:
        """Add a box mesh and return the instance index."""
        vertices, indices = self._create_box_geometry(min_pt, max_pt)
        mesh = Mesh(vertices, indices, material_id=material_id)
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def add_sphere(
        self,
        center: tuple,
        radius: float,
        segments: int,
        material_id: int,
    ) -> int:
        """
        Add a UV sphere mesh and return the instance index.

        Args:
            center: Sphere center
            radius: Sphere radius
            segments: Horizontal tessellation (minimum 8)
            material_id: Material index
        """
        vertices, indices, normals, texcoords = self._create_sphere_geometry(
            center=center,
            radius=radius,
            segments=max(8, int(segments)),
        )
        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            texcoords=texcoords,
            material_id=material_id,
        )
        mesh_idx = self.add_mesh(mesh)
        return self.add_instance(mesh_idx)

    def _create_box_geometry(self, min_pt: tuple, max_pt: tuple) -> tuple:
        """Create box vertices and indices."""
        mn = np.array(min_pt, dtype=np.float32)
        mx = np.array(max_pt, dtype=np.float32)

        # 8 corners
        c = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ],
            dtype=np.float32,
        )

        # 6 faces, 4 vertices each (with normals)
        vertices = []
        indices = []

        def add_quad(v0, v1, v2, v3, normal):
            base = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

        # Front (+Z)
        add_quad(c[4], c[5], c[6], c[7], [0, 0, 1])
        # Back (-Z)
        add_quad(c[1], c[0], c[3], c[2], [0, 0, -1])
        # Left (-X)
        add_quad(c[0], c[4], c[7], c[3], [-1, 0, 0])
        # Right (+X)
        add_quad(c[5], c[1], c[2], c[6], [1, 0, 0])
        # Bottom (-Y)
        add_quad(c[0], c[1], c[5], c[4], [0, -1, 0])
        # Top (+Y)
        add_quad(c[7], c[6], c[2], c[3], [0, 1, 0])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def _create_sphere_geometry(
        self,
        center: tuple,
        radius: float,
        segments: int,
    ) -> tuple:
        """Create UV sphere vertices, indices, normals, and UVs."""
        cx, cy, cz = center
        lat_segments = max(4, segments // 2)
        lon_segments = max(8, segments)

        vertices = []
        normals = []
        texcoords = []
        indices = []

        for y in range(lat_segments + 1):
            v = y / lat_segments
            theta = v * np.pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for x in range(lon_segments + 1):
                u = x / lon_segments
                phi = u * (2.0 * np.pi)
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                nx = sin_theta * cos_phi
                ny = cos_theta
                nz = sin_theta * sin_phi

                px = cx + radius * nx
                py = cy + radius * ny
                pz = cz + radius * nz

                vertices.append([px, py, pz])
                normals.append([nx, ny, nz])
                texcoords.append([u, 1.0 - v])

        stride = lon_segments + 1
        for y in range(lat_segments):
            for x in range(lon_segments):
                i0 = y * stride + x
                i1 = i0 + 1
                i2 = i0 + stride
                i3 = i2 + 1
                indices.append([i0, i2, i1])
                indices.append([i1, i2, i3])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
            np.array(texcoords, dtype=np.float32),
        )

    def clear(self):
        """Clear all meshes and instances."""
        self._meshes.clear()
        self._instances.clear()
        self._sphere_lights.clear()
        self._sphere_light_data = None
        self._gas_handles.clear()
        self._gas_buffers.clear()
        self._ias_handle = None
        self._ias_buffer = None
        self._instance_buffer = None
        self._instance_record_floats = None
        self._instance_record_words = None
        self._render_node_floats = None
        self._device_instance_transforms = None
        self._instance_records_dirty = True
        self._instance_np_cache = None
        self._instance_np_capacity = 0
        self._instance_transform_cache = np.empty((0, 4, 4), dtype=np.float32)
        self._instance_visibility_cache = np.empty(0, dtype=np.bool_)
        self._instance_cache_capacity = 0
        self._tlas_temp_buffer = None
        self._tlas_temp_capacity = 0
        self._tlas_output_capacity = 0
        self._tlas_output_size = 0
        self._tlas_instance_count = 0
        self._keepalive.clear()
        self.materials.clear()
        self._instance_material_ids = None
        self._compact_materials = None
        self._compact_material_floats = None
        self._instance_render_prim_ids = None
        self._texture_data = None
        self._texture_objects = []
        self._packed_indices = None
        self._packed_normals = None
        self._packed_tangents = None
        self._packed_texcoords0 = None
        self._packed_texcoords1 = None
        self._packed_prev_positions = None
        self._packed_material_ids = None
        self._gltf_textures = []
        self.usd_environment_path = None
        self.usd_ambient_light = (0.0, 0.0, 0.0)
        self.usd_scene = None
        self._usd_node_parents = np.empty(0, dtype=np.int32)
        self._usd_local_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_world_transforms = np.empty((0, 4, 4), dtype=np.float32)
        self._usd_instance_ids = np.empty(0, dtype=np.int32)
        self._usd_instance_node_ids = np.empty(0, dtype=np.int32)
        self._usd_level_nodes = []
        self._d_usd_node_parents = None
        self._d_usd_local_transforms = None
        self._d_usd_world_transforms = None
        self._d_usd_instance_ids = None
        self._d_usd_instance_node_ids = None
        self._d_usd_level_nodes = []

    def build(self, optix_module):
        """
        Build the scene: upload meshes, create BLASes, and create TLAS.

        Args:
            optix_module: The imported optix module
        """
        self._optix = optix_module

        if len(self._meshes) == 0:
            logger.info("No meshes to build.")
            return

        # A full build may be requested again after prototypes or instances
        # are added. Drop the previous acceleration-structure references so
        # the mesh and instance indices stay aligned.
        self._gas_handles.clear()
        self._gas_buffers.clear()
        self._ias_handle = None
        self._tlas_instance_count = 0
        self._keepalive.clear()
        self._instance_records_dirty = True

        logger.info(
            "Building scene with %d meshes and %d instances.",
            len(self._meshes),
            len(self._instances),
        )

        # Upload meshes to GPU
        for mesh in self._meshes:
            mesh.upload_to_gpu()

        # Build BLAS for each mesh
        self._build_blas()

        # Build TLAS
        self._build_tlas()

        # Build scene buffers
        self._build_scene_buffers()

        total_verts = sum(len(m.vertices) for m in self._meshes)
        total_tris = sum(len(m.indices) for m in self._meshes)
        logger.info(
            "Scene build complete: %d vertices, %d triangles.", total_verts, total_tris
        )

    def rebuild_tlas(self):
        """Rebuild only TLAS after instance transform updates."""
        if len(self._meshes) == 0 or len(self._instances) == 0:
            return
        if self._optix is None:
            return
        if len(self._gas_handles) != len(self._meshes):
            # Fallback if a full build has not been completed yet.
            self.build(self._optix)
            return
        self._build_tlas()

    def _build_blas(self):
        """Build bottom-level acceleration structures for all meshes."""
        optix = self._optix

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        stream = int(wp.get_stream("cuda").cuda_stream)
        build_inputs = []
        max_temp_size = 0
        for mesh in self._meshes:
            tri = optix.BuildInputTriangleArray()
            tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
            tri.numVertices = len(mesh.vertices)
            tri.vertexStrideInBytes = 12
            tri.vertexBuffers = [mesh.d_vertices.ptr]
            tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
            tri.numIndexTriplets = len(mesh.indices)
            tri.indexStrideInBytes = 12
            tri.indexBuffer = mesh.d_indices.ptr
            tri.flags = [optix.GEOMETRY_FLAG_NONE]
            tri.numSbtRecords = 1

            sizes = self._ctx.accelComputeMemoryUsage([accel_options], [tri])
            build_inputs.append((tri, sizes))
            max_temp_size = max(max_temp_size, int(sizes.tempSizeInBytes))

        # Every build is submitted to the same CUDA stream, so its scratch
        # range can be reused once the preceding build reaches completion.
        # Large composed USD stages otherwise retain hundreds of independent
        # temporary allocations until the full build is synchronized.
        d_temp = wp.empty(max_temp_size, dtype=wp.uint8, device="cuda")
        for tri, sizes in build_inputs:
            d_gas = wp.empty(sizes.outputSizeInBytes, dtype=wp.uint8, device="cuda")

            handle = self._ctx.accelBuild(
                stream,
                [accel_options],
                [tri],
                d_temp.ptr,
                sizes.tempSizeInBytes,
                d_gas.ptr,
                sizes.outputSizeInBytes,
                [],
            )

            self._gas_handles.append(int(handle))
            self._gas_buffers.append(d_gas)
        self._keepalive["gas_temp"] = d_temp

    def _build_tlas(self):
        """Build top-level acceleration structure."""
        optix = self._optix

        if len(self._instances) == 0:
            return
        inst_dtype = _build_optix_instance_dtype()
        count = len(self._instances)
        if self._instance_np_capacity < count:
            self._instance_np_capacity = max(count, 1, self._instance_np_capacity * 2)
            self._instance_np_cache = np.zeros(
                self._instance_np_capacity, dtype=inst_dtype
            )
        required_bytes = count * inst_dtype.itemsize
        buffer_changed = (
            self._instance_buffer is None
            or self._instance_buffer.shape[0] != required_bytes
        )
        if buffer_changed:
            self._instance_buffer = wp.empty(
                required_bytes, dtype=wp.uint8, device="cuda"
            )
            self._instance_record_floats = None
            self._instance_record_words = None
            self._instance_records_dirty = True
        if self._instance_records_dirty:
            inst_np = self._instance_np_cache[:count]
            mesh_indices = np.fromiter(
                (inst.mesh_index for inst in self._instances),
                dtype=np.intp,
                count=count,
            )
            inst_np["transform"] = self._instance_transform_cache[
                :count, :3, :
            ].reshape(count, 12)
            inst_np["instanceId"] = np.arange(count, dtype=np.uint32)
            inst_np["sbtOffset"] = np.uint32(0)
            inst_np["visibilityMask"] = np.where(
                self._instance_visibility_cache[:count], 0xFF, 0
            ).astype(np.uint32)
            inst_np["flags"] = np.fromiter(
                (
                    int(optix.INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING)
                    if inst.double_sided
                    else int(optix.INSTANCE_FLAG_NONE)
                    for inst in self._instances
                ),
                dtype=np.uint32,
                count=count,
            )
            inst_np["traversableHandle"] = np.asarray(
                self._gas_handles, dtype=np.uint64
            )[mesh_indices]
            self._instance_buffer.assign(inst_np.view(np.uint8).reshape(-1))
            self._instance_records_dirty = False

        build_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_UPDATE),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        ias_input = optix.BuildInputInstanceArray()
        ias_input.instances = int(self._instance_buffer.ptr)
        ias_input.numInstances = count

        sizes = self._ctx.accelComputeMemoryUsage([build_options], [ias_input])
        required_build_temp = int(sizes.tempSizeInBytes)
        required_update_temp = int(sizes.tempUpdateSizeInBytes)
        required_output = int(sizes.outputSizeInBytes)

        can_update = (
            self._ias_handle is not None
            and not buffer_changed
            and self._tlas_instance_count == count
            and self._ias_buffer is not None
            and self._tlas_output_size == required_output
        )

        # Reuse TLAS scratch/output buffers across rebuilds.
        # Overallocate slightly to reduce realloc churn when scene size
        # changes by a small amount.
        output_buffer_changed = (
            self._ias_buffer is None or self._tlas_output_capacity < required_output
        )
        if output_buffer_changed:
            self._tlas_output_capacity = max(
                required_output, int(required_output * 1.25)
            )
            self._ias_buffer = wp.empty(
                self._tlas_output_capacity, dtype=wp.uint8, device="cuda"
            )
            can_update = False

        required_temp = required_update_temp if can_update else required_build_temp
        if self._tlas_temp_buffer is None or self._tlas_temp_capacity < required_temp:
            self._tlas_temp_capacity = max(required_temp, int(required_temp * 1.25))
            self._tlas_temp_buffer = wp.empty(
                self._tlas_temp_capacity, dtype=wp.uint8, device="cuda"
            )

        accel_options = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_ALLOW_UPDATE),
            operation=optix.BUILD_OPERATION_UPDATE
            if can_update
            else optix.BUILD_OPERATION_BUILD,
        )

        self._ias_handle = self._ctx.accelBuild(
            int(wp.get_stream("cuda").cuda_stream),
            [accel_options],
            [ias_input],
            self._tlas_temp_buffer.ptr,
            required_temp,
            self._ias_buffer.ptr,
            required_output,
            [],
        )

        self._tlas_output_size = required_output
        self._tlas_instance_count = count
        self._keepalive["ias_temp"] = self._tlas_temp_buffer

    def _build_scene_buffers(self):
        """Build GPU buffers for scene description."""
        # Build RenderPrimitive array
        rp_dtype = _create_render_primitive_dtype()
        render_primitives = np.zeros(len(self._meshes), dtype=rp_dtype)
        packed_indices = []
        packed_normals = []
        packed_tangents = []
        packed_texcoords0 = []
        packed_texcoords1 = []
        packed_material_ids = []

        index_offset = 0
        normal_offset = 0
        tangent_offset = 0
        tex0_offset = 0
        tex1_offset = 0
        material_offset = 0

        for i, mesh in enumerate(self._meshes):
            rp = render_primitives[i]
            flat_indices = mesh.indices.reshape(-1)
            flat_normals = mesh.normals.reshape(-1)
            flat_tangents = mesh.tangents.reshape(-1)
            flat_tex0 = mesh.texcoords.reshape(-1)
            flat_tex1 = mesh.texcoords1.reshape(-1)
            tri_material_ids = np.full(
                len(mesh.indices), mesh.material_id, dtype=np.uint32
            )

            rp["indexOffset"] = np.uint32(index_offset)
            rp["materialIdOffset"] = np.uint32(material_offset)
            rp["vertexBuffer"]["positionOffset"] = np.uint32(0)
            rp["vertexBuffer"]["normalOffset"] = np.uint32(normal_offset)
            rp["vertexBuffer"]["colorOffset"] = np.uint32(0)
            rp["vertexBuffer"]["tangentOffset"] = np.uint32(tangent_offset)
            rp["vertexBuffer"]["texCoord0Offset"] = np.uint32(tex0_offset)
            rp["vertexBuffer"]["texCoord1Offset"] = np.uint32(tex1_offset)
            rp["vertexBuffer"]["prevPositionOffset"] = np.uint32(0)
            rp["vertexBuffer"]["hasTexCoord1"] = np.uint32(1)
            rp["vertexBuffer"]["hasPrevPosition"] = np.uint32(0)
            rp["numIndices"] = np.uint32(mesh.indices.size)
            rp["numVertices"] = np.uint32(mesh.vertices.shape[0])

            packed_indices.append(flat_indices.astype(np.uint32, copy=False))
            packed_normals.append(flat_normals.astype(np.float32, copy=False))
            packed_tangents.append(flat_tangents.astype(np.float32, copy=False))
            packed_texcoords0.append(flat_tex0.astype(np.float32, copy=False))
            packed_texcoords1.append(flat_tex1.astype(np.float32, copy=False))
            packed_material_ids.append(tri_material_ids)

            index_offset += flat_indices.shape[0]
            normal_offset += flat_normals.shape[0]
            tangent_offset += flat_tangents.shape[0]
            tex0_offset += flat_tex0.shape[0]
            tex1_offset += flat_tex1.shape[0]
            material_offset += tri_material_ids.shape[0]

        rp_bytes = render_primitives.view(np.uint8).reshape(-1)
        self._render_primitives = wp.array(rp_bytes, dtype=wp.uint8, device="cuda")
        self._packed_indices = wp.array(
            np.concatenate(packed_indices), dtype=wp.uint32, device="cuda"
        )
        self._packed_normals = wp.array(
            np.concatenate(packed_normals), dtype=wp.float32, device="cuda"
        )
        self._packed_tangents = wp.array(
            np.concatenate(packed_tangents), dtype=wp.float32, device="cuda"
        )
        self._packed_texcoords0 = wp.array(
            np.concatenate(packed_texcoords0), dtype=wp.float32, device="cuda"
        )
        self._packed_texcoords1 = wp.array(
            np.concatenate(packed_texcoords1), dtype=wp.float32, device="cuda"
        )
        self._packed_prev_positions = None
        self._packed_material_ids = wp.array(
            np.concatenate(packed_material_ids), dtype=wp.uint32, device="cuda"
        )

        # Build RenderNode array
        rn_dtype = _create_render_node_dtype()
        render_nodes = np.zeros(len(self._instances), dtype=rn_dtype)

        count = len(self._instances)
        transforms = self._instance_transform_cache[:count]
        render_nodes["objectToWorld"] = transforms
        render_nodes["worldToObject"] = np.linalg.inv(transforms)
        render_nodes["materialID"] = -1  # Use per-triangle materials
        render_nodes["renderPrimID"] = np.fromiter(
            (inst.mesh_index for inst in self._instances), dtype=np.int32, count=count
        )

        rn_bytes = render_nodes.view(np.uint8).reshape(-1)
        self._render_nodes = wp.array(rn_bytes, dtype=wp.uint8, device="cuda")

        count = len(self._instances)
        record_capacity = int(self._instance_buffer.capacity)
        self._instance_record_floats = wp.array(
            ptr=self._instance_buffer.ptr,
            shape=(count, 20),
            dtype=wp.float32,
            capacity=record_capacity,
            device="cuda",
        )
        self._instance_record_words = wp.array(
            ptr=self._instance_buffer.ptr,
            shape=(count, 20),
            dtype=wp.uint32,
            capacity=record_capacity,
            device="cuda",
        )
        self._render_node_floats = wp.array(
            ptr=self._render_nodes.ptr,
            shape=(count, 34),
            dtype=wp.float32,
            capacity=int(self._render_nodes.capacity),
            device="cuda",
        )
        transforms = self._instance_transform_cache[:count, :3, :].reshape(count, 12)
        self._device_instance_transforms = wp.array(
            transforms, dtype=wp.float32, device="cuda"
        )
        self._build_usd_transform_buffers()

        light_data = np.zeros((len(self._sphere_lights), 8), dtype=np.float32)
        for index, (position, radius, color, intensity) in enumerate(
            self._sphere_lights
        ):
            light_data[index, :3] = position
            light_data[index, 3] = radius
            light_data[index, 4:7] = np.asarray(color) * intensity
        self._sphere_light_data = (
            wp.array(light_data, dtype=wp.float32, device="cuda")
            if len(light_data)
            else None
        )

        # Build SceneDescription
        sd_dtype = _create_scene_description_dtype()
        scene_desc = np.zeros(1, dtype=sd_dtype)
        scene_desc[0]["materialAddress"] = self.materials.gpu_address
        scene_desc[0]["renderNodeAddress"] = self._render_nodes.ptr
        scene_desc[0]["renderPrimitiveAddress"] = self._render_primitives.ptr
        scene_desc[0]["lightAddress"] = (
            0 if self._sphere_light_data is None else self._sphere_light_data.ptr
        )
        scene_desc[0]["numLights"] = len(self._sphere_lights)

        sd_bytes = scene_desc.view(np.uint8).reshape(-1)
        self._scene_desc = wp.array(sd_bytes, dtype=wp.uint8, device="cuda")

        # Build per-instance material ID lookup buffer.
        instance_material_ids = np.zeros(len(self._instances), dtype=np.uint32)
        instance_render_prim_ids = np.zeros(len(self._instances), dtype=np.uint32)
        for i, inst in enumerate(self._instances):
            material_id = (
                self._meshes[inst.mesh_index].material_id
                if inst.material_id is None
                else inst.material_id
            )
            instance_material_ids[i] = np.uint32(material_id)
            instance_render_prim_ids[i] = np.uint32(inst.mesh_index)
        self._instance_material_ids = wp.array(
            instance_material_ids, dtype=wp.uint32, device="cuda"
        )
        self._instance_render_prim_ids = wp.array(
            instance_render_prim_ids, dtype=wp.uint32, device="cuda"
        )

        # Build compact material table for robust device-side lookup.
        compact_dt = np.dtype(
            [
                ("baseColor", np.float32, (3,)),
                ("emissive", np.float32, (3,)),
                ("roughness", np.float32),
                ("metallic", np.float32),
                ("uSubdiv", np.float32),
                ("vSubdiv", np.float32),
                ("baseColorScale", np.float32),
                ("baseColorAdd", np.float32),
                ("baseColorDesaturation", np.float32),
                ("alphaMode", np.int32),
                ("alphaCutoff", np.float32),
                ("transmission", np.float32),
                ("ior", np.float32),
                ("specularColor", np.float32, (3,)),
                ("specular", np.float32),
                ("clearcoat", np.float32),
                ("clearcoatRoughness", np.float32),
                ("sheenRoughness", np.float32),
                ("occlusion", np.float32),
                ("occlusionTexIndex", np.int32),
                ("occlusionTexCoord", np.int32),
                ("sheenColor", np.float32, (3,)),
                ("diffuseTransmissionFactor", np.float32),
                ("diffuseTransmissionColor", np.float32, (3,)),
                ("isThinWalled", np.int32),
                ("clearcoatNormalTexIndex", np.int32),
                ("clearcoatNormalTexCoord", np.int32),
                ("opacity", np.float32),
                ("baseColorTexIndex", np.int32),
                ("baseColorTexCoord", np.int32),
                ("metallicRoughnessTexIndex", np.int32),
                ("metallicRoughnessTexCoord", np.int32),
                ("normalTexIndex", np.int32),
                ("normalTexCoord", np.int32),
                ("emissiveTexIndex", np.int32),
                ("emissiveTexCoord", np.int32),
                ("normalScale", np.float32),
                ("baseColorUvTransform", np.float32, (6,)),
                ("metallicRoughnessUvTransform", np.float32, (6,)),
                ("normalUvTransform", np.float32, (6,)),
                ("emissiveUvTransform", np.float32, (6,)),
                ("occlusionUvTransform", np.float32, (6,)),
                ("clearcoatNormalUvTransform", np.float32, (6,)),
            ],
            align=True,
        )
        compact = np.zeros(self.materials.count, dtype=compact_dt)
        for i, mat in enumerate(self.materials._materials):
            compact[i]["baseColor"] = mat["pbrBaseColorFactor"][:3]
            compact[i]["emissive"] = mat["emissiveFactor"]
            compact[i]["roughness"] = mat["pbrRoughnessFactor"]
            compact[i]["metallic"] = mat["pbrMetallicFactor"]
            compact[i]["uSubdiv"] = mat["uSubdiv"]
            compact[i]["vSubdiv"] = mat["vSubdiv"]
            compact[i]["baseColorScale"] = mat["baseColorScale"]
            compact[i]["baseColorAdd"] = mat["pbrDiffuseFactor"][0]
            compact[i]["baseColorDesaturation"] = mat["pbrDiffuseFactor"][1]
            compact[i]["alphaMode"] = mat["alphaMode"]
            compact[i]["alphaCutoff"] = mat["alphaCutoff"]
            compact[i]["transmission"] = mat["transmissionFactor"]
            compact[i]["ior"] = mat["ior"]
            compact[i]["specularColor"] = mat["specularColorFactor"]
            compact[i]["specular"] = mat["specularFactor"]
            compact[i]["clearcoat"] = mat["clearcoatFactor"]
            compact[i]["clearcoatRoughness"] = mat["clearcoatRoughness"]
            compact[i]["sheenRoughness"] = mat["sheenRoughnessFactor"]
            compact[i]["occlusion"] = mat["occlusionStrength"]
            compact[i]["occlusionTexIndex"] = mat["occlusionTexture"]["index"]
            compact[i]["occlusionTexCoord"] = mat["occlusionTexture"]["texCoord"]
            compact[i]["sheenColor"] = mat["sheenColorFactor"]
            compact[i]["diffuseTransmissionFactor"] = mat["diffuseTransmissionFactor"]
            compact[i]["diffuseTransmissionColor"] = mat["diffuseTransmissionColor"]
            compact[i]["isThinWalled"] = 1 if mat["thicknessFactor"] == 0.0 else 0
            compact[i]["clearcoatNormalTexIndex"] = mat["clearcoatNormalTexture"][
                "index"
            ]
            compact[i]["clearcoatNormalTexCoord"] = mat["clearcoatNormalTexture"][
                "texCoord"
            ]
            compact[i]["opacity"] = mat["pbrBaseColorFactor"][3]
            compact[i]["baseColorTexIndex"] = mat["pbrBaseColorTexture"]["index"]
            compact[i]["baseColorTexCoord"] = mat["pbrBaseColorTexture"]["texCoord"]
            compact[i]["metallicRoughnessTexIndex"] = mat[
                "pbrMetallicRoughnessTexture"
            ]["index"]
            compact[i]["metallicRoughnessTexCoord"] = mat[
                "pbrMetallicRoughnessTexture"
            ]["texCoord"]
            compact[i]["normalTexIndex"] = mat["normalTexture"]["index"]
            compact[i]["normalTexCoord"] = mat["normalTexture"]["texCoord"]
            compact[i]["emissiveTexIndex"] = mat["emissiveTexture"]["index"]
            compact[i]["emissiveTexCoord"] = mat["emissiveTexture"]["texCoord"]
            compact[i]["normalScale"] = mat["normalTextureScale"]
            compact[i]["baseColorUvTransform"] = (
                mat["pbrBaseColorTexture"]["uvTransform00"],
                mat["pbrBaseColorTexture"]["uvTransform01"],
                mat["pbrBaseColorTexture"]["uvTransform02"],
                mat["pbrBaseColorTexture"]["uvTransform10"],
                mat["pbrBaseColorTexture"]["uvTransform11"],
                mat["pbrBaseColorTexture"]["uvTransform12"],
            )
            compact[i]["metallicRoughnessUvTransform"] = (
                mat["pbrMetallicRoughnessTexture"]["uvTransform00"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform01"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform02"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform10"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform11"],
                mat["pbrMetallicRoughnessTexture"]["uvTransform12"],
            )
            compact[i]["normalUvTransform"] = (
                mat["normalTexture"]["uvTransform00"],
                mat["normalTexture"]["uvTransform01"],
                mat["normalTexture"]["uvTransform02"],
                mat["normalTexture"]["uvTransform10"],
                mat["normalTexture"]["uvTransform11"],
                mat["normalTexture"]["uvTransform12"],
            )
            compact[i]["emissiveUvTransform"] = (
                mat["emissiveTexture"]["uvTransform00"],
                mat["emissiveTexture"]["uvTransform01"],
                mat["emissiveTexture"]["uvTransform02"],
                mat["emissiveTexture"]["uvTransform10"],
                mat["emissiveTexture"]["uvTransform11"],
                mat["emissiveTexture"]["uvTransform12"],
            )
            compact[i]["occlusionUvTransform"] = (
                mat["occlusionTexture"]["uvTransform00"],
                mat["occlusionTexture"]["uvTransform01"],
                mat["occlusionTexture"]["uvTransform02"],
                mat["occlusionTexture"]["uvTransform10"],
                mat["occlusionTexture"]["uvTransform11"],
                mat["occlusionTexture"]["uvTransform12"],
            )
            compact[i]["clearcoatNormalUvTransform"] = (
                mat["clearcoatNormalTexture"]["uvTransform00"],
                mat["clearcoatNormalTexture"]["uvTransform01"],
                mat["clearcoatNormalTexture"]["uvTransform02"],
                mat["clearcoatNormalTexture"]["uvTransform10"],
                mat["clearcoatNormalTexture"]["uvTransform11"],
                mat["clearcoatNormalTexture"]["uvTransform12"],
            )
        compact_bytes = compact.view(np.uint8).reshape(-1)
        self._compact_materials = wp.array(compact_bytes, dtype=wp.uint8, device="cuda")
        self._compact_material_floats = None

        # Keep one CUDA texture per source image. The compact device array stores
        # only texture handles, so aggregate texture memory is not constrained by
        # Warp's signed 32-bit limit for a single array dimension.
        if self._gltf_textures:
            self._texture_objects = [
                wp.Texture2D(
                    tex,
                    filter_mode=wp.TextureFilterMode.LINEAR,
                    address_mode=wp.TextureAddressMode.WRAP,
                    normalized_coords=True,
                    device="cuda",
                )
                for tex in self._gltf_textures
            ]
            self._texture_data = wp.array(
                self._texture_objects,
                dtype=wp.Texture2D,
                device="cuda",
            )
        else:
            self._texture_data = None
            self._texture_objects = []
