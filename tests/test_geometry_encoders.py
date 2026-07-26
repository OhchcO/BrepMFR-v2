import pytest
import torch

from models.modules.cylindrical_encoder import CylindricalPositionEncoder
from models.modules.pointcloud_encoder import PointCloudFaceEncoder


def valid_face_point_batch(face_count=3):
    points = torch.randn(face_count, 96, 6)
    mask = torch.zeros(face_count, 96, dtype=torch.bool)
    mask[:, :32] = True
    points[:, :32, 3:] = torch.nn.functional.normalize(
        points[:, :32, 3:], dim=-1
    )
    return points, mask


def test_point_encoder_ignores_padded_values():
    encoder = PointCloudFaceEncoder(hidden_dim=128).eval()
    points = torch.randn(2, 96, 6)
    mask = torch.zeros(2, 96, dtype=torch.bool)
    mask[:, :32] = True
    changed = points.clone()
    changed[:, 32:] = 1e6

    with torch.no_grad():
        assert torch.allclose(encoder(points, mask), encoder(changed, mask))


def test_encoders_return_finite_hidden_vectors_and_point_gradients():
    points, mask = valid_face_point_batch(face_count=3)
    points.requires_grad_()
    encoded_points = PointCloudFaceEncoder(128)(points, mask)
    encoded_cylindrical = CylindricalPositionEncoder(128)(torch.randn(3, 4))

    assert encoded_points.shape == (3, 128)
    assert torch.isfinite(encoded_points).all()
    assert encoded_cylindrical.shape == (3, 128)
    assert torch.isfinite(encoded_cylindrical).all()
    encoded_points.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_encoders_reject_malformed_inputs():
    points, mask = valid_face_point_batch(face_count=1)
    with pytest.raises(ValueError, match="face_points"):
        PointCloudFaceEncoder(16)(points[..., :5], mask)
    with pytest.raises(ValueError, match="zero valid"):
        PointCloudFaceEncoder(16)(points, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="finite"):
        CylindricalPositionEncoder(16)(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]))
