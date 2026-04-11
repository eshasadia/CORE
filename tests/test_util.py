"""
Unit tests for displacement field and TRE utility functions.
"""
import numpy as np
import pytest

# Skip this entire module if the heavy optional dependencies are not installed.
util = pytest.importorskip("core.utils.util",
                           reason="core.utils.util requires torch and tiatoolbox")
tre = util.tre
rtre = util.rtre
combine_deformation = util.combine_deformation
apply_deformation_to_points = util.apply_deformation_to_points


class TestTREUtilFunctions:
    """Tests that directly call the low-level TRE functions in util.py."""

    def test_tre_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = tre(pts, pts)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_tre_known_values(self):
        pts1 = np.array([[0.0, 0.0], [0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0], [5.0, 12.0]])
        result = tre(pts1, pts2)
        np.testing.assert_array_almost_equal(result, [5.0, 13.0])

    def test_rtre_normalisation(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])   # distance = 5; diagonal of 3×4 image = 5
        result = rtre(pts1, pts2, x_size=3, y_size=4)
        np.testing.assert_array_almost_equal(result, [1.0])


class TestCombineDeformation:
    """Verify that composing two identity displacement fields yields an identity."""

    def _zero_field(self, h, w):
        return np.zeros((h, w), dtype=np.float32)

    def test_two_zeros_give_zero(self):
        h, w = 20, 30
        u_x = self._zero_field(h, w)
        u_y = self._zero_field(h, w)
        out_x, out_y = combine_deformation(u_x, u_y, u_x, u_y)
        np.testing.assert_array_almost_equal(out_x, 0.0, decimal=5)
        np.testing.assert_array_almost_equal(out_y, 0.0, decimal=5)

    def test_output_shape(self):
        h, w = 15, 25
        u = self._zero_field(h, w)
        out_x, out_y = combine_deformation(u, u, u, u)
        assert out_x.shape == (h, w)
        assert out_y.shape == (h, w)


class TestApplyDeformationToPoints:
    def test_zero_field_no_movement(self):
        """A zero deformation field should not move any point."""
        h, w = 50, 50
        deformation_field = np.zeros((2, h, w), dtype=np.float32)
        points = np.array([[10.0, 10.0], [20.0, 30.0], [40.0, 5.0]])
        result = apply_deformation_to_points(points, deformation_field)
        np.testing.assert_array_almost_equal(result, points)

    def test_uniform_translation(self):
        """A uniform displacement of (+3, +5) should shift all points accordingly."""
        h, w = 60, 60
        deformation_field = np.zeros((2, h, w), dtype=np.float32)
        deformation_field[0] = 3.0   # dx
        deformation_field[1] = 5.0   # dy
        points = np.array([[10.0, 10.0]])
        result = apply_deformation_to_points(points, deformation_field)
        np.testing.assert_array_almost_equal(result, [[13.0, 15.0]])

    def test_output_shape(self):
        h, w = 40, 40
        df = np.zeros((2, h, w), dtype=np.float32)
        n_pts = 7
        pts = np.random.rand(n_pts, 2) * 30  # keep within field bounds
        result = apply_deformation_to_points(pts, df)
        assert result.shape == (n_pts, 2)
