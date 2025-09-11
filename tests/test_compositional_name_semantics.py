import pytest
import numpy as np
import pandas as pd
from vtools.functions.merge import ts_merge, ts_splice

class TestNamesSemantics:
    def _univariate_inputs(self):
        # Daily, aligned indexes to avoid reindexing edge-cases
        idx = pd.date_range("2023-01-01", periods=5, freq="D")
        s1 = pd.Series([1, 2, 3, 4, 5], index=idx, name="A")
        # lower priority, only fills last two points
        s2 = pd.Series([np.nan, np.nan, np.nan, 40, 50], index=idx, name="B")
        return s1, s2

    # --- ts_merge ---

    def test_merge_univariate_names_str_is_univariate(self):
        s1, s2 = self._univariate_inputs()
        out = ts_merge((s1, s2), names="C")
        assert isinstance(out, pd.Series), "Expected a univariate Series when names is a string"
        assert out.name == "C"

    def test_merge_univariate_names_list_single_equals_str(self):
        s1, s2 = self._univariate_inputs()
        out_str  = ts_merge((s1, s2), names="C")
        out_list = ts_merge((s1, s2), names=["C"])
        # Both should be Series named 'C' and identical in values/index
        assert isinstance(out_list, pd.Series)
        assert out_list.name == "C"
        pd.testing.assert_series_equal(out_list, out_str)

    def test_merge_univariate_names_list_multi_raises(self):
        s1, s2 = self._univariate_inputs()
        with pytest.raises(ValueError, match="multiple names"):
            ts_merge((s1, s2), names=["X", "Y"])

    # --- ts_splice ---

    def test_splice_univariate_names_str_is_univariate(self):
        s1, s2 = self._univariate_inputs()
        out = ts_splice((s1, s2), names="C", transition="prefer_last")
        assert isinstance(out, pd.Series), "Expected a univariate Series when names is a string"
        assert out.name == "C"

    def test_splice_univariate_names_list_single_equals_str(self):
        s1, s2 = self._univariate_inputs()
        out_str  = ts_splice((s1, s2), names="C", transition="prefer_last")
        out_list = ts_splice((s1, s2), names=["C"], transition="prefer_last")
        assert isinstance(out_list, pd.Series)
        assert out_list.name == "C"
        pd.testing.assert_series_equal(out_list, out_str)

    def test_splice_univariate_names_list_multi_raises(self):
        s1, s2 = self._univariate_inputs()
        with pytest.raises(ValueError, match="multiple names"):
            ts_splice((s1, s2), names=["X", "Y"], transition="prefer_last")
