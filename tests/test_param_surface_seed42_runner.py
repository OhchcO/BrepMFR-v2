from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_mfrinstseg_param_surface_seed42.ps1"
)


def test_seed42_param_surface_runner_has_the_screening_matrix():
    text = RUNNER.read_text(encoding="utf-8")

    assert "MFRInstSeg_BrepMFR_ParamSurface_v4_seed42" in text
    assert '$Seed = 42' in text
    assert text.count('Mode = "pc_aag"') == 5
    assert text.count('NeighborMetric = "intrinsic"') == 4
    assert text.count('NeighborMetric = "xyz"') == 1
    assert text.count('TrimAware = $false') == 3
    assert text.count('TrimAware = $true') == 2
    assert text.count('Fusion = "add"') == 3
    assert text.count('Fusion = "gated"') == 2
    assert 'ResultLabel = "pc_aag_param_surface_xyz_untrimmed_add"' in text
    assert (
        'ResultLabel = "pc_aag_param_surface_intrinsic_untrimmed_gated"'
        in text
    )


def test_seed42_param_surface_runner_trains_and_tests_fresh_checkpoints():
    text = RUNNER.read_text(encoding="utf-8")

    assert "run_mfrinstseg_experiments.ps1" in text
    assert "run_mfrinstseg_test.ps1" in text
    assert "ExistingCheckpointPaths" in text
    assert "No fresh best checkpoint found" in text
    assert "RebuildPreparedData -and $Index -eq 0" in text


def test_seed42_param_surface_runner_skips_completed_results_when_resuming():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'Filter "test_log.txt"' in text
    assert "Skipping completed experiment" in text
    assert "-not $RebuildPreparedData" in text
    assert "continue" in text
