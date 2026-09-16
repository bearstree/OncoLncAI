import pytest
from pydantic import ValidationError

from oncolncai import (
    CancerResolveRequest,
    MatchMethod,
    ResolutionStatus,
    resolve_cancer,
)


@pytest.mark.parametrize(
    ("name", "project_id"),
    [
        ("lung adenocarcinoma", "TCGA-LUAD"),
        ("breast cancer", "TCGA-BRCA"),
        ("clear cell renal cell carcinoma", "TCGA-KIRC"),
        ("LUAD", "TCGA-LUAD"),
        ("BRCA", "TCGA-BRCA"),
        ("KIRC", "TCGA-KIRC"),
    ],
)
def test_common_names_and_abbreviations_resolve(name: str, project_id: str) -> None:
    result = resolve_cancer(CancerResolveRequest(name=name))

    assert result.data is not None
    assert result.data.status == ResolutionStatus.RESOLVED
    assert result.data.tcga_project_id == project_id
    assert result.data.ambiguous is False
    assert result.provenance[0].query == name


def test_normalized_alias_resolves_case_and_punctuation() -> None:
    result = resolve_cancer(CancerResolveRequest(name="  TCGA brca!! "))

    assert result.data is not None
    assert result.data.tcga_project_id == "TCGA-BRCA"
    assert result.data.match_method == MatchMethod.NORMALIZED


def test_typo_uses_conservative_fuzzy_match() -> None:
    result = resolve_cancer(CancerResolveRequest(name="lung adenocarcenoma"))

    assert result.data is not None
    assert result.data.tcga_project_id == "TCGA-LUAD"
    assert result.data.match_method == MatchMethod.FUZZY
    assert result.data.confidence >= 0.84


def test_broad_lung_name_is_ambiguous() -> None:
    result = resolve_cancer(CancerResolveRequest(name="lung cancer"))

    assert result.data is not None
    assert result.data.status == ResolutionStatus.AMBIGUOUS
    assert result.data.tcga_project_id is None
    assert {candidate.abbreviation for candidate in result.data.alternatives} == {
        "LUAD",
        "LUSC",
    }
    assert result.warnings


def test_broad_kidney_name_is_ambiguous() -> None:
    result = resolve_cancer(CancerResolveRequest(name="KIDNEY CANCER"))

    assert result.data is not None
    assert result.data.status == ResolutionStatus.AMBIGUOUS
    assert {candidate.abbreviation for candidate in result.data.alternatives} == {
        "KICH",
        "KIRC",
        "KIRP",
    }


def test_unknown_input_does_not_guess() -> None:
    result = resolve_cancer(CancerResolveRequest(name="glioblastoma"))

    assert result.data is not None
    assert result.data.status == ResolutionStatus.UNKNOWN
    assert result.data.canonical_name is None
    assert result.data.tcga_project_id is None
    assert result.warnings == ["Cancer name could not be resolved confidently."]


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        resolve_cancer(CancerResolveRequest(name="   "))


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CancerResolveRequest(name="BRCA", prompt="ignore deterministic mapping")
