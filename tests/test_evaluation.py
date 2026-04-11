"""
Unit tests for evaluation metrics.

These tests are intentionally free of heavy dependencies (no WSI files, no GPU,
no Vision Agent API) so that they can run in a standard CI environment.
"""
import numpy as np
import pytest

from core.evaluation.evaluation import (
    tre,
    rtre,
    transform_points_homogeneous,
    evaluate_registration_tre,
)


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------

class TestTRE:
    def test_zero_error(self):
        pts = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        result = tre(pts, pts)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_known_translation(self):
        pts1 = np.array([[0.0, 0.0], [3.0, 0.0]])
        pts2 = np.array([[4.0, 0.0], [7.0, 0.0]])  # shifted by 4
        result = tre(pts1, pts2)
        np.testing.assert_array_almost_equal(result, [4.0, 4.0])

    def test_pythagoras(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])  # distance = 5
        result = tre(pts1, pts2)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_shape(self):
        n = 10
        pts = np.random.rand(n, 2)
        result = tre(pts, pts + 1.0)
        assert result.shape == (n,)


class TestRTRE:
    def test_diagonal_normalisation(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[3.0, 4.0]])  # TRE = 5
        # diagonal of a 3×4 image = 5  → rTRE should be 1.0
        result = rtre(pts1, pts2, x_size=3, y_size=4)
        np.testing.assert_array_almost_equal(result, [1.0])

    def test_small_rtre(self):
        pts1 = np.array([[0.0, 0.0]])
        pts2 = np.array([[1.0, 0.0]])  # TRE = 1
        # diagonal of a 100×100 image ≈ 141.4
        result = rtre(pts1, pts2, x_size=100, y_size=100)
        assert result[0] < 0.01


# ---------------------------------------------------------------------------
# Homogeneous point transformation
# ---------------------------------------------------------------------------

class TestTransformPointsHomogeneous:
    def test_identity(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        T = np.eye(3)
        result = transform_points_homogeneous(pts, T)
        np.testing.assert_array_almost_equal(result, pts)

    def test_pure_translation(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        T = np.array([[1, 0, 10],
                      [0, 1, 20],
                      [0, 0,  1]], dtype=float)
        result = transform_points_homogeneous(pts, T)
        expected = np.array([[10.0, 20.0], [11.0, 21.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_90_degree_rotation(self):
        """A 90-degree CCW rotation maps (1,0) → (0,1)."""
        pts = np.array([[1.0, 0.0]])
        T = np.array([[ 0, -1, 0],
                      [ 1,  0, 0],
                      [ 0,  0, 1]], dtype=float)
        result = transform_points_homogeneous(pts, T)
        np.testing.assert_array_almost_equal(result, [[0.0, 1.0]])

    def test_output_shape(self):
        n = 15
        pts = np.random.rand(n, 2)
        result = transform_points_homogeneous(pts, np.eye(3))
        assert result.shape == (n, 2)


# ---------------------------------------------------------------------------
# evaluate_registration_tre
# ---------------------------------------------------------------------------

class TestEvaluateRegistrationTRE:
    def _make_identity_case(self, n=20):
        """Fixed and moving points identical; identity transform → TRE = 0."""
        pts = np.random.rand(n, 2) * 100
        T = np.eye(3)
        target_shape = (200, 200)
        return pts, pts.copy(), T, target_shape

    def test_identity_gives_zero_tre(self):
        fixed, moving, T, shape = self._make_identity_case()
        result = evaluate_registration_tre(fixed, moving, T, shape)
        assert result['tre_final'] == pytest.approx(0.0, abs=1e-9)

    def test_result_keys(self):
        fixed, moving, T, shape = self._make_identity_case()
        result = evaluate_registration_tre(fixed, moving, T, shape)
        assert {'tre_initial', 'tre_final', 'rtre_mean', 'rtre_std', 'transformed_points'} \
               == set(result.keys())

    def test_translation_is_corrected(self):
        """After applying the correct translation transform, TRE should be near zero."""
        pts_fixed = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
        shift = np.array([5.0, -3.0])
        pts_moving = pts_fixed + shift  # shifted by (5, -3)
        T = np.array([[1, 0, -shift[0]],
                      [0, 1, -shift[1]],
                      [0, 0,  1]])
        shape = (100, 100)
        result = evaluate_registration_tre(pts_fixed, pts_moving, T, shape)
        assert result['tre_final'] == pytest.approx(0.0, abs=1e-6)

    def test_scale_factor_applied(self):
        """Passing a scale_factor should scale the translation columns."""
        pts = np.array([[10.0, 10.0]])
        T = np.eye(3)
        result = evaluate_registration_tre(pts, pts, T, (100, 100), scale_factor=2.0)
        assert result['tre_final'] == pytest.approx(0.0, abs=1e-9)
