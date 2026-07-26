import pathlib

import pytest

from models.brepseg_model import feature_result_path as brepseg_result_path
from models.transfer_model import feature_result_path as transfer_result_path


@pytest.mark.parametrize("result_path", [brepseg_result_path, transfer_result_path])
def test_test_time_result_paths_accept_full_stem_string_ids(tmp_path, result_path):
    assert result_path(tmp_path, "part-alpha_001") == (
        pathlib.Path(tmp_path) / "feature_part-alpha_001.txt"
    )


@pytest.mark.parametrize("result_path", [brepseg_result_path, transfer_result_path])
@pytest.mark.parametrize("sample_id", ["", ".", "..", "nested/sample"])
def test_test_time_result_paths_reject_unsafe_string_ids(result_path, sample_id):
    with pytest.raises(ValueError, match="sample ID"):
        result_path("results", sample_id)
