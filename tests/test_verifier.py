from oncolncai import (
    ClaimKind,
    EvidenceReference,
    MockLLMProvider,
    NumericAssertion,
    Provenance,
    ResultStatus,
    VerificationClaim,
    VerificationRequest,
    verify_claims,
)


SOURCE_SENTENCE = "MALAT1 expression was associated with shorter overall survival in this cohort."


def reference(**changes: object) -> EvidenceReference:
    values: dict[str, object] = {
        "reference_id": "survival:luad:malat1",
        "source_type": "survival_analysis",
        "source_text": SOURCE_SENTENCE,
        "gene_identifier": "MALAT1",
        "numeric_values": {"sample_count": 120.0, "hazard_ratio": 1.75, "cox_p_value": 0.012},
        "provenance": (
            Provenance(source="TCGA", source_id="TCGA-LUAD", dataset_version="fixture-v1"),
        ),
    }
    values.update(changes)
    return EvidenceReference.model_validate(values)


def claim(**changes: object) -> VerificationClaim:
    values: dict[str, object] = {
        "claim_id": "claim-1",
        "text": SOURCE_SENTENCE,
        "kind": ClaimKind.SURVIVAL,
        "gene_identifier": "MALAT1",
        "evidence_reference_ids": ("survival:luad:malat1",),
        "numeric_assertions": (
            NumericAssertion(
                evidence_reference_id="survival:luad:malat1",
                metric="hazard_ratio",
                reported_value=1.75,
            ),
        ),
    }
    values.update(changes)
    return VerificationClaim.model_validate(values)


def test_clean_exact_claim_passes_without_llm() -> None:
    provider = MockLLMProvider([])
    result = verify_claims(
        VerificationRequest(claims=(claim(),), evidence=(reference(),)),
        semantic_provider=provider,
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None and result.data.passed is True
    assert provider.calls == []
    assert result.provenance[0].source_id == "TCGA-LUAD"


def test_numeric_mismatch_and_absent_metric_are_flagged_deterministically() -> None:
    assertions = (
        NumericAssertion(
            evidence_reference_id="survival:luad:malat1", metric="hazard_ratio", reported_value=2.4
        ),
        NumericAssertion(
            evidence_reference_id="survival:luad:malat1", metric="log_rank_p_value", reported_value=0.03
        ),
    )
    provider = MockLLMProvider([])
    result = verify_claims(
        VerificationRequest(claims=(claim(numeric_assertions=assertions),), evidence=(reference(),)),
        semantic_provider=provider,
    )

    assert result.data is not None and result.data.passed is False
    assert len(result.data.numeric_mismatches) == 2
    assert result.data.numeric_mismatches[0].source_value == 1.75
    assert "absent" in result.data.numeric_mismatches[1].message
    assert provider.calls == []


def test_source_identifier_provenance_and_causal_guards_all_run() -> None:
    bad_claim = claim(
        kind=ClaimKind.LITERATURE,
        text="MALAT1 causes poor survival.",
        gene_identifier="not a valid identifier!",
        pmid=None,
        evidence_reference_ids=("missing-reference",),
        numeric_assertions=(),
    )
    result = verify_claims(VerificationRequest(claims=(bad_claim,), evidence=(reference(),)))

    assert result.data is not None and result.data.passed is False
    assert len(result.data.missing_sources) == 2
    assert len(result.data.invalid_identifiers) == 1
    assert len(result.data.causal_language_flags) == 1

    no_provenance = verify_claims(
        VerificationRequest(
            claims=(claim(numeric_assertions=()),),
            evidence=(reference(provenance=()),),
        )
    )
    assert no_provenance.data is not None
    assert len(no_provenance.data.missing_provenance) == 1


def test_valid_literature_pmid_is_required_to_match_linked_source() -> None:
    literature_reference = reference(
        reference_id="pubmed:12345678",
        source_type="PubMed",
        pmid="12345678",
        provenance=(Provenance(source="NCBI PubMed", source_id="12345678"),),
    )
    literature_claim = claim(
        kind=ClaimKind.LITERATURE,
        pmid="87654321",
        evidence_reference_ids=("pubmed:12345678",),
        numeric_assertions=(),
    )

    result = verify_claims(
        VerificationRequest(claims=(literature_claim,), evidence=(literature_reference,))
    )

    assert result.data is not None
    assert result.data.missing_sources[0].message == "claim PMID does not match a linked source"


def test_ambiguous_clean_claim_uses_semantic_provider_once() -> None:
    paraphrase = claim(text="Higher MALAT1 was linked to worse overall survival.")
    provider = MockLLMProvider(
        [{"decisions": [{"claim_id": "claim-1", "supported": True, "reason": "Faithful paraphrase."}]}]
    )

    result = verify_claims(
        VerificationRequest(claims=(paraphrase,), evidence=(reference(),)),
        semantic_provider=provider,
    )

    assert result.data is not None and result.data.passed is True
    assert len(result.data.semantic_checks) == 1
    assert len(provider.calls) == 1
    assert provider.calls[0][1].__name__ == "SemanticSupportBatch"


def test_semantically_unsupported_claim_is_reported() -> None:
    provider = MockLLMProvider(
        [{"decisions": [{"claim_id": "claim-1", "supported": False, "reason": "The source does not mention response to therapy."}]}]
    )
    result = verify_claims(
        VerificationRequest(
            claims=(claim(text="MALAT1 predicts response to therapy."),),
            evidence=(reference(),),
        ),
        semantic_provider=provider,
    )

    assert result.data is not None and result.data.passed is False
    assert result.data.unsupported_claims[0].claim_id == "claim-1"


def test_missing_semantic_provider_fails_closed_with_warning() -> None:
    result = verify_claims(
        VerificationRequest(
            claims=(claim(text="Higher MALAT1 was linked to worse survival."),),
            evidence=(reference(),),
        )
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.data is not None and result.data.passed is False
    assert len(result.data.unsupported_claims) == 1
    assert result.warnings == ["1 claim(s) could not be semantically verified."]


def test_invalid_semantic_output_is_a_structured_failure() -> None:
    provider = MockLLMProvider([{"decisions": []}])
    result = verify_claims(
        VerificationRequest(
            claims=(claim(text="Higher MALAT1 was linked to worse survival."),),
            evidence=(reference(),),
        ),
        semantic_provider=provider,
    )

    assert result.status == ResultStatus.FAILURE
    assert result.error is not None
    assert result.error.code == "invalid_semantic_verification"
