"""Deterministic-first scientific claim verification and guardrails."""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from oncolncai.providers import LLMProvider
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


class ClaimKind(StrEnum):
    LITERATURE = "literature"
    SURVIVAL = "survival"
    DIFFERENTIAL_EXPRESSION = "differential_expression"


class NumericAssertion(BaseModel):
    """A number printed in a claim and the evidence field it should equal."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    evidence_reference_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    reported_value: float


class VerificationClaim(BaseModel):
    """A scientific statement and its structured citations."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: ClaimKind
    gene_identifier: str | None = None
    pmid: str | None = None
    evidence_reference_ids: tuple[str, ...] = ()
    numeric_assertions: tuple[NumericAssertion, ...] = ()


class EvidenceReference(BaseModel):
    """Small verifier-facing view of a source or deterministic tool result."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    reference_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_text: str | None = None
    gene_identifier: str | None = None
    pmid: str | None = None
    numeric_values: dict[str, float] = Field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()

    @model_validator(mode="after")
    def validate_numeric_values(self) -> EvidenceReference:
        if any(not math.isfinite(value) for value in self.numeric_values.values()):
            raise ValueError("numeric_values must be finite")
        return self


class VerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: tuple[VerificationClaim, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = ()
    relative_tolerance: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-9, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> VerificationRequest:
        claim_ids = [claim.claim_id for claim in self.claims]
        reference_ids = [reference.reference_id for reference in self.evidence]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference_id values must be unique")
        return self


class VerificationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    message: str


class NumericMismatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    evidence_reference_id: str
    metric: str
    reported_value: float
    source_value: float | None = None
    message: str


class CausalLanguageFlag(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    matched_text: str
    message: str


class SemanticSupportDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    supported: bool
    reason: str = Field(min_length=1)


class SemanticSupportBatch(BaseModel):
    """Schema-constrained output for the optional semantic verification step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decisions: tuple[SemanticSupportDecision, ...]


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    unsupported_claims: tuple[VerificationIssue, ...] = ()
    numeric_mismatches: tuple[NumericMismatch, ...] = ()
    missing_sources: tuple[VerificationIssue, ...] = ()
    invalid_identifiers: tuple[VerificationIssue, ...] = ()
    missing_provenance: tuple[VerificationIssue, ...] = ()
    causal_language_flags: tuple[CausalLanguageFlag, ...] = ()
    semantic_checks: tuple[SemanticSupportDecision, ...] = ()
    warnings: tuple[str, ...] = ()


_ENSEMBL_GENE = re.compile(r"^ENSG\d{11}(?:\.\d+)?$", re.IGNORECASE)
_GENE_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{0,30}$")
_CAUSAL_PATTERNS = (
    re.compile(r"\bcaus(?:e|es|ed|ing)\b", re.IGNORECASE),
    re.compile(r"\b(?:lead|leads|led) to\b", re.IGNORECASE),
    re.compile(r"\bresult(?:s|ed|ing)? in\b", re.IGNORECASE),
    re.compile(r"\bdriv(?:e|es|en|ing)\b", re.IGNORECASE),
    re.compile(r"\bprevent(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bproves?\b", re.IGNORECASE),
)


def _valid_gene_identifier(value: str) -> bool:
    return bool(_ENSEMBL_GENE.fullmatch(value) or _GENE_SYMBOL.fullmatch(value))


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _collect_provenance(evidence: tuple[EvidenceReference, ...]) -> list[Provenance]:
    collected: list[Provenance] = []
    seen: set[str] = set()
    for reference in evidence:
        for provenance in reference.provenance:
            key = json.dumps(provenance.model_dump(mode="json"), sort_keys=True)
            if key not in seen:
                seen.add(key)
                collected.append(provenance)
    return collected


def _semantic_prompt(
    candidates: list[tuple[VerificationClaim, list[EvidenceReference]]],
) -> str:
    payload = [
        {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "sources": [
                {
                    "reference_id": reference.reference_id,
                    "source_text": reference.source_text,
                }
                for reference in references
            ],
        }
        for claim, references in candidates
    ]
    return (
        "Determine whether each claim is explicitly supported by its supplied source text. "
        "Do not infer causation, invent facts, or alter numerical results. Return one decision "
        "for every claim_id.\n" + json.dumps(payload, ensure_ascii=False)
    )


def verify_claims(
    request: VerificationRequest,
    *,
    semantic_provider: LLMProvider | None = None,
) -> ToolResult[VerificationReport]:
    """Verify claims with deterministic checks before optional semantic review."""

    references = {reference.reference_id: reference for reference in request.evidence}
    missing_sources: list[VerificationIssue] = []
    invalid_identifiers: list[VerificationIssue] = []
    missing_provenance: list[VerificationIssue] = []
    numeric_mismatches: list[NumericMismatch] = []
    causal_flags: list[CausalLanguageFlag] = []
    blocked_claim_ids: set[str] = set()

    for claim in request.claims:
        linked = [references[ref] for ref in claim.evidence_reference_ids if ref in references]
        unknown = [ref for ref in claim.evidence_reference_ids if ref not in references]
        if not claim.evidence_reference_ids:
            missing_sources.append(VerificationIssue(claim_id=claim.claim_id, message="claim has no source reference"))
        for reference_id in unknown:
            missing_sources.append(
                VerificationIssue(claim_id=claim.claim_id, message=f"unknown source reference: {reference_id}")
            )
        if claim.kind == ClaimKind.LITERATURE:
            if claim.pmid is None or not claim.pmid.isdigit():
                missing_sources.append(
                    VerificationIssue(claim_id=claim.claim_id, message="literature claim has no valid PMID")
                )
            elif linked and not any(reference.pmid == claim.pmid for reference in linked):
                missing_sources.append(
                    VerificationIssue(claim_id=claim.claim_id, message="claim PMID does not match a linked source")
                )
        if claim.gene_identifier is None or not _valid_gene_identifier(claim.gene_identifier):
            invalid_identifiers.append(
                VerificationIssue(claim_id=claim.claim_id, message="missing or invalid gene/lncRNA identifier")
            )
        elif any(
            reference.gene_identifier
            and reference.gene_identifier.casefold() != claim.gene_identifier.casefold()
            for reference in linked
        ):
            invalid_identifiers.append(
                VerificationIssue(claim_id=claim.claim_id, message="gene/lncRNA identifier does not match linked evidence")
            )
        for reference in linked:
            if not reference.provenance:
                missing_provenance.append(
                    VerificationIssue(
                        claim_id=claim.claim_id,
                        message=f"source {reference.reference_id} has no provenance",
                    )
                )
        for pattern in _CAUSAL_PATTERNS:
            match = pattern.search(claim.text)
            if match:
                causal_flags.append(
                    CausalLanguageFlag(
                        claim_id=claim.claim_id,
                        matched_text=match.group(0),
                        message="causal wording is prohibited for observational evidence",
                    )
                )
        for assertion in claim.numeric_assertions:
            reference = references.get(assertion.evidence_reference_id)
            source_value = None if reference is None else reference.numeric_values.get(assertion.metric)
            if reference is None or assertion.evidence_reference_id not in claim.evidence_reference_ids:
                message = "numeric assertion does not point to a linked source"
            elif source_value is None:
                message = "numeric metric is absent from the linked source"
            elif math.isclose(
                assertion.reported_value,
                source_value,
                rel_tol=request.relative_tolerance,
                abs_tol=request.absolute_tolerance,
            ):
                continue
            else:
                message = "reported value does not match the deterministic source value"
            numeric_mismatches.append(
                NumericMismatch(
                    claim_id=claim.claim_id,
                    evidence_reference_id=assertion.evidence_reference_id,
                    metric=assertion.metric,
                    reported_value=assertion.reported_value,
                    source_value=source_value,
                    message=message,
                )
            )

    for issue in (*missing_sources, *invalid_identifiers, *missing_provenance, *numeric_mismatches, *causal_flags):
        blocked_claim_ids.add(issue.claim_id)

    candidates: list[tuple[VerificationClaim, list[EvidenceReference]]] = []
    for claim in request.claims:
        if claim.claim_id in blocked_claim_ids:
            continue
        linked = [references[ref] for ref in claim.evidence_reference_ids]
        texts = [reference.source_text for reference in linked if reference.source_text]
        if texts and any(_normalized_text(claim.text) in _normalized_text(text) for text in texts):
            continue
        else:
            candidates.append((claim, linked))

    semantic_checks: tuple[SemanticSupportDecision, ...] = ()
    unsupported: list[VerificationIssue] = []
    warnings: list[str] = []
    if candidates and semantic_provider is None:
        unsupported.extend(
            VerificationIssue(
                claim_id=claim.claim_id,
                message="claim support requires semantic review but no provider was supplied",
            )
            for claim, _ in candidates
        )
        warnings.append(f"{len(candidates)} claim(s) could not be semantically verified.")
    elif candidates:
        try:
            raw = semantic_provider.generate_structured(
                prompt=_semantic_prompt(candidates), output_schema=SemanticSupportBatch
            )
            batch = SemanticSupportBatch.model_validate(raw)
            expected = {claim.claim_id for claim, _ in candidates}
            returned = [decision.claim_id for decision in batch.decisions]
            if len(returned) != len(set(returned)) or set(returned) != expected:
                raise ValueError("semantic decisions must contain each requested claim_id exactly once")
            semantic_checks = batch.decisions
        except (ValidationError, ValueError) as exc:
            return ToolResult[VerificationReport](
                status=ResultStatus.FAILURE,
                error=ToolError(code="invalid_semantic_verification", message=str(exc), retryable=False),
                provenance=_collect_provenance(request.evidence),
            )
        except Exception as exc:
            return ToolResult[VerificationReport](
                status=ResultStatus.FAILURE,
                error=ToolError(
                    code="semantic_verifier_failed",
                    message=str(exc) or type(exc).__name__,
                    retryable=True,
                ),
                provenance=_collect_provenance(request.evidence),
            )
        unsupported.extend(
            VerificationIssue(claim_id=decision.claim_id, message=decision.reason)
            for decision in semantic_checks
            if not decision.supported
        )

    passed = not any(
        (
            unsupported,
            numeric_mismatches,
            missing_sources,
            invalid_identifiers,
            missing_provenance,
            causal_flags,
        )
    )
    report = VerificationReport(
        passed=passed,
        unsupported_claims=tuple(unsupported),
        numeric_mismatches=tuple(numeric_mismatches),
        missing_sources=tuple(missing_sources),
        invalid_identifiers=tuple(invalid_identifiers),
        missing_provenance=tuple(missing_provenance),
        causal_language_flags=tuple(causal_flags),
        semantic_checks=semantic_checks,
        warnings=tuple(warnings),
    )
    return ToolResult[VerificationReport](
        status=ResultStatus.SUCCESS,
        data=report,
        provenance=_collect_provenance(request.evidence),
        warnings=list(warnings),
    )
