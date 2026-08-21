import numpy as np

from warp_optix.pathtracing.scene import Mesh


def test_mesh_generates_area_weighted_normals_without_authored_normals():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)

    mesh = Mesh(vertices, indices)

    np.testing.assert_allclose(mesh.normals, [[0.0, 0.0, 1.0]] * 3)


def test_mesh_without_uvs_uses_deterministic_fallback_tangents():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2]], dtype=np.uint32)

    mesh = Mesh(vertices, indices)

    np.testing.assert_array_equal(
        mesh.tangents,
        np.array([[1.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32),
    )
