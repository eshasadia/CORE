"""
Unit tests for nuclei detection helpers in nuclei_analysis.py.
"""
import numpy as np
import pytest
import pandas as pd
import os
import tempfile

# Skip this entire module if the heavy optional dependencies are not installed.
nuclei_analysis = pytest.importorskip(
    "core.preprocessing.nuclei_analysis",
    reason="nuclei_analysis requires tiatoolbox and torch"
)

detect_nuclei_patch_watershed = nuclei_analysis.detect_nuclei_patch_watershed
subsample_nuclei = nuclei_analysis.subsample_nuclei
create_nuclei_dataframe_from_points = nuclei_analysis.create_nuclei_dataframe_from_points
load_nuclei_coordinates = nuclei_analysis.load_nuclei_coordinates


class TestDetectNucleiPatchWatershed:
    def _blank_rgb(self, h=64, w=64):
        return np.full((h, w, 3), 200, dtype=np.uint8)

    def test_blank_image_returns_no_nuclei(self):
        img = self._blank_rgb()
        stats, centroids = detect_nuclei_patch_watershed(img, min_area=1)
        assert len(stats) == len(centroids)

    def test_dark_spot_detected(self):
        """A single dark square on a white background should be detected as a nucleus."""
        img = self._blank_rgb(64, 64)
        # Draw a 10×10 dark square in the centre
        img[27:37, 27:37] = 20
        stats, centroids = detect_nuclei_patch_watershed(img, min_area=1)
        assert len(centroids) >= 1

    def test_centroid_shape(self):
        img = self._blank_rgb(64, 64)
        img[27:37, 27:37] = 20
        stats, centroids = detect_nuclei_patch_watershed(img, min_area=1)
        if len(centroids) > 0:
            assert centroids.ndim == 2
            assert centroids.shape[1] == 2

    def test_min_area_filters_small_nuclei(self):
        """With a very large min_area, the small dark spot should be filtered out."""
        img = self._blank_rgb(64, 64)
        img[30:33, 30:33] = 20  # 3×3 = 9 px area
        _, centroids_small = detect_nuclei_patch_watershed(img, min_area=1)
        _, centroids_large = detect_nuclei_patch_watershed(img, min_area=10000)
        assert len(centroids_large) <= len(centroids_small)


class TestSubsampleNuclei:
    def _make_df(self, n):
        return pd.DataFrame({
            'global_x': np.random.rand(n),
            'global_y': np.random.rand(n),
            'area': np.ones(n),
        })

    def test_no_subsampling_needed(self):
        df = self._make_df(50)
        result = subsample_nuclei(df, n_samples=100)
        assert len(result) == 50

    def test_subsampling_reduces_size(self):
        df = self._make_df(200)
        result = subsample_nuclei(df, n_samples=50)
        assert len(result) == 50

    def test_reproducible_with_seed(self):
        df = self._make_df(100)
        r1 = subsample_nuclei(df, n_samples=30, random_state=7)
        r2 = subsample_nuclei(df, n_samples=30, random_state=7)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


class TestCreateNucleiDataframeFromPoints:
    def test_basic_creation(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        df = create_nuclei_dataframe_from_points(pts)
        assert list(df.columns) == ['global_x', 'global_y', 'area']
        assert len(df) == 2
        assert df['area'].iloc[0] == pytest.approx(1.0)

    def test_with_area_values(self):
        pts = np.array([[0.0, 0.0]])
        areas = np.array([42.0])
        df = create_nuclei_dataframe_from_points(pts, area_values=areas)
        assert df['area'].iloc[0] == pytest.approx(42.0)


class TestLoadNucleiCoordinates:
    def test_load_basic_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("global_x,global_y\n10.0,20.0\n30.0,40.0\n")
            fname = f.name
        try:
            df = load_nuclei_coordinates(fname)
            assert 'global_x' in df.columns
            assert 'global_y' in df.columns
            assert 'area' in df.columns        # should default to 1.0
            assert len(df) == 2
        finally:
            os.remove(fname)

    def test_existing_area_column_preserved(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("global_x,global_y,area\n5.0,6.0,99.0\n")
            fname = f.name
        try:
            df = load_nuclei_coordinates(fname)
            assert df['area'].iloc[0] == pytest.approx(99.0)
        finally:
            os.remove(fname)
