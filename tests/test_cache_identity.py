import pytest

from oncolncai.cache_identity import (
    CacheCompatibilityError, RealCacheIdentity, cache_identity_hash,
    validate_cached_real_artifact,
)


def _identity(**updates) -> RealCacheIdentity:
    values = {
        "project_id": "TCGA-LUAD", "source_dataset_id": "GDC STAR Counts",
        "source_dataset_checksum_or_version": "sha256:abc", "expression_representation": "raw_integer_counts",
        "analysis_configuration": {"design": "group", "fdr": 0.05}, "de_method": "PYDESEQ2",
        "statistical_thresholds": {"fdr": 0.05, "absolute_log2_fc": 1.0},
        "annotation_version": "GENCODE-v36", "project_version": "0.1.0",
        "software_versions": {"python": "3.12", "pydeseq2": "0.5"}, "source_git_sha": "abc123",
    }
    values.update(updates)
    return RealCacheIdentity(**values)


def test_identity_hash_is_stable_and_project_specific() -> None:
    expected = "21456cafc18b0ff3d444e33de844d3abbda7229231edbefcc7f6402451c711f0"
    assert cache_identity_hash(_identity()) == expected
    assert cache_identity_hash(_identity(project_id="TCGA-BRCA")) != expected


@pytest.mark.parametrize("field,value", [
    ("annotation_version", "GENCODE-v44"),
    ("source_dataset_checksum_or_version", "sha256:different"),
    ("analysis_configuration", {"design": "group", "fdr": 0.10}),
    ("source_git_sha", "different"),
])
def test_incompatible_real_cache_is_rejected(field, value) -> None:
    with pytest.raises(CacheCompatibilityError, match=field):
        validate_cached_real_artifact(_identity(**{field: value}), _identity())


def test_compatible_real_cache_is_explicitly_labeled_reused() -> None:
    result = validate_cached_real_artifact(_identity(), _identity())
    assert result == "REUSED_VALIDATED_REAL_ARTIFACT"
