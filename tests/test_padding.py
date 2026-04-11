"""
Unit tests for image padding utilities.
"""
import numpy as np
import pytest

from core.preprocessing.padding import (
    calculate_pad_value,
    pad_images,
    apply_padding_landmarks,
    pad_landmarks,
)


class TestCalculatePadValue:
    def test_same_size(self):
        pad1, pad2 = calculate_pad_value((100, 100), (100, 100))
        assert pad1 == [(0, 0), (0, 0)]
        assert pad2 == [(0, 0), (0, 0)]

    def test_image1_taller(self):
        """image_1 taller → image_2 should receive vertical padding."""
        pad1, pad2 = calculate_pad_value((110, 100), (100, 100))
        assert pad1[0] == (0, 0)        # no padding on image_1 height
        assert sum(pad2[0]) == 10       # 10 px total vertical padding on image_2

    def test_image2_wider(self):
        """image_2 wider → image_1 should receive horizontal padding."""
        pad1, pad2 = calculate_pad_value((100, 80), (100, 100))
        assert sum(pad1[1]) == 20       # 20 px total horizontal padding on image_1
        assert pad2[1] == (0, 0)

    def test_symmetric_pad(self):
        """Even size difference should result in equal top/bottom pads."""
        pad1, pad2 = calculate_pad_value((100, 100), (90, 100))
        # image_2 is 10 px shorter → image_2 height pad = (5, 5)
        assert pad2[0] == (5, 5)

    def test_asymmetric_odd(self):
        """Odd size difference → floor goes to leading side, ceil to trailing."""
        pad1, pad2 = calculate_pad_value((100, 100), (91, 100))
        total = pad2[0][0] + pad2[0][1]
        assert total == 9


class TestPadImages:
    def _make_rgb(self, h, w):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_same_size_unchanged(self):
        img1 = self._make_rgb(50, 60)
        img2 = self._make_rgb(50, 60)
        out1, out2, params = pad_images(img1, img2)
        assert out1.shape == out2.shape == (50, 60, 3)

    def test_different_heights(self):
        img1 = self._make_rgb(80, 60)
        img2 = self._make_rgb(60, 60)
        out1, out2, params = pad_images(img1, img2)
        assert out1.shape == out2.shape

    def test_different_widths(self):
        img1 = self._make_rgb(60, 100)
        img2 = self._make_rgb(60, 80)
        out1, out2, params = pad_images(img1, img2)
        assert out1.shape == out2.shape

    def test_both_dimensions_differ(self):
        img1 = self._make_rgb(70, 90)
        img2 = self._make_rgb(50, 100)
        out1, out2, _ = pad_images(img1, img2)
        assert out1.shape[:2] == out2.shape[:2]
        assert out1.shape[0] == max(70, 50)
        assert out1.shape[1] == max(90, 100)

    def test_padding_params_returned(self):
        img1 = self._make_rgb(60, 60)
        img2 = self._make_rgb(80, 80)
        _, _, params = pad_images(img1, img2)
        assert 'pad_1' in params
        assert 'pad_2' in params


class TestApplyPaddingLandmarks:
    def test_no_padding(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        pad = [(0, 0), (0, 0)]
        result = apply_padding_landmarks(pts, pad)
        np.testing.assert_array_equal(result, pts)

    def test_x_padding(self):
        pts = np.array([[10.0, 20.0]])
        pad = [(0, 0), (5, 5)]   # 5 px left padding
        result = apply_padding_landmarks(pts, pad)
        assert result[0, 0] == pytest.approx(15.0)   # x shifted by 5

    def test_y_padding(self):
        pts = np.array([[10.0, 20.0]])
        pad = [(3, 3), (0, 0)]   # 3 px top padding
        result = apply_padding_landmarks(pts, pad)
        assert result[0, 1] == pytest.approx(23.0)   # y shifted by 3

    def test_original_unmodified(self):
        pts = np.array([[10.0, 20.0]])
        pad = [(2, 2), (3, 3)]
        original = pts.copy()
        apply_padding_landmarks(pts, pad)
        np.testing.assert_array_equal(pts, original)


class TestPadLandmarks:
    def test_both_sets_padded(self):
        pts1 = np.array([[5.0, 5.0]])
        pts2 = np.array([[10.0, 10.0]])
        params = {
            'pad_1': [(2, 2), (3, 3)],
            'pad_2': [(1, 1), (4, 4)],
        }
        r1, r2 = pad_landmarks(params, pts1, pts2)
        assert r1[0, 0] == pytest.approx(8.0)   # x + 3 (pad_1 x leading)
        assert r1[0, 1] == pytest.approx(7.0)   # y + 2 (pad_1 y leading)
        assert r2[0, 0] == pytest.approx(14.0)  # x + 4 (pad_2 x leading)
        assert r2[0, 1] == pytest.approx(11.0)  # y + 1 (pad_2 y leading)

    def test_none_passthrough(self):
        params = {'pad_1': [(0, 0), (0, 0)], 'pad_2': [(0, 0), (0, 0)]}
        r1, r2 = pad_landmarks(params, None, None)
        assert r1 is None
        assert r2 is None
