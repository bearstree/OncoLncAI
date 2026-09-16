"""Bounded retrieval and deterministic parsing of standardized GEO matrices."""

from __future__ import annotations

import csv
import gzip
import io
import re
import json
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from oncolncai.geo_replication import GEOExpressionFeature, GEOExpressionSample, GEOReplicationRequest


class GEOIntegrationStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    INTEGRATION_INCOMPLETE = "INTEGRATION_INCOMPLETE"


class GEOTumorNormal(StrEnum):
    TUMOR = "tumor"
    NORMAL = "normal"
    UNKNOWN = "unknown"


class GEOSamplePhenotype(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    disease: str | None = None
    subtype: str | None = None
    tissue: str | None = None
    tumor_normal: GEOTumorNormal = GEOTumorNormal.UNKNOWN
    survival_time: float | None = None
    survival_event: int | None = Field(default=None, ge=0, le=1)
    treatment: str | None = None
    platform: str | None = None
    raw_metadata: tuple[str, ...] = ()


class GEOSeriesMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str
    platform: str
    sample_ids: tuple[str, ...]
    sample_titles: tuple[str, ...]
    source_names: tuple[str, ...]
    characteristics: tuple[tuple[str, ...], ...]
    feature_ids: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]
    source_url: str


class GEOPlatformMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    platform: str
    feature_to_symbols: dict[str, tuple[str, ...]]
    source_url: str
    total_features: int
    mapped_features: int
    ambiguous_features: int


class GEOCompatibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str
    status: GEOIntegrationStatus
    reasons: tuple[str, ...]
    tumor_count: int
    normal_count: int
    candidate_coverage: tuple[str, ...]
    phenotype_quality: float = Field(ge=0, le=1)
    survival_available: bool


class GEOParsedCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    matrix: GEOSeriesMatrix
    phenotypes: tuple[GEOSamplePhenotype, ...]
    mapping: GEOPlatformMapping
    compatibility: GEOCompatibilityResult
    replication_request: GEOReplicationRequest | None = None


class GEORetrievalTransport(Protocol):
    def get_bytes(self, url: str, timeout: float) -> bytes: ...


class UrllibGEORetrievalTransport:
    def get_bytes(self, url: str, timeout: float) -> bytes:
        with urlopen(Request(url, headers={"User-Agent": "OncoLncAI/0.1"}), timeout=timeout) as response:  # noqa: S310
            return response.read()


class PhenotypeRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tumor_patterns: tuple[str, ...] = (r"\btumou?r\b", r"\bcancer\b", r"\bcarcinoma\b")
    normal_patterns: tuple[str, ...] = (r"\bnormal\b", r"non[- ]tumou?r", r"adjacent non[- ]tumou?r")
    disease_patterns: tuple[str, ...] = ()
    tissue_patterns: tuple[str, ...] = ()


class CuratedGEOCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accession: str = Field(pattern=r"^GSE\d+$")
    cancer: str
    subtype: str | None = None
    tissue: str
    platform: str = Field(pattern=r"^GPL\d+$")
    sample_group_field: str
    tumor_values: tuple[str, ...]
    normal_values: tuple[str, ...]
    survival_time_field: str | None = None
    survival_event_field: str | None = None
    known_identifier_mapping: str | None = None
    verification_status: str = Field(pattern=r"^(verified|draft|rejected)$")
    notes: str
    provenance: str


class CuratedGEORegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    cohorts: tuple[CuratedGEOCohort, ...] = ()

    def verified(self, accession: str, platform: str) -> CuratedGEOCohort | None:
        return next((item for item in self.cohorts if item.accession == accession and item.platform == platform and item.verification_status == "verified"), None)


def load_curated_geo_registry(path: Path | None = None) -> CuratedGEORegistry:
    source = path or Path(__file__).with_name("data") / "geo_cohorts.json"
    return CuratedGEORegistry.model_validate_json(source.read_text(encoding="utf-8"))


def apply_curated_phenotypes(matrix: GEOSeriesMatrix, phenotypes: tuple[GEOSamplePhenotype, ...],
                             cohort: CuratedGEOCohort) -> tuple[GEOSamplePhenotype, ...]:
    if cohort.verification_status != "verified" or cohort.accession != matrix.accession or cohort.platform != matrix.platform:
        raise ValueError("curated phenotype mapping is not verified for this accession/platform")
    tumor_values = {item.casefold() for item in cohort.tumor_values}
    normal_values = {item.casefold() for item in cohort.normal_values}
    resolved = []
    for phenotype in phenotypes:
        observed = _metadata_value(phenotype.raw_metadata, (cohort.sample_group_field.casefold(),))
        if observed is None:
            resolved.append(phenotype)
            continue
        value = observed.casefold()
        label = GEOTumorNormal.TUMOR if value in tumor_values else GEOTumorNormal.NORMAL if value in normal_values else GEOTumorNormal.UNKNOWN
        resolved.append(phenotype.model_copy(update={
            "tumor_normal": label, "disease": phenotype.disease or cohort.cancer,
            "subtype": phenotype.subtype or cohort.subtype, "tissue": phenotype.tissue or cohort.tissue,
        }))
    return tuple(resolved)


def geo_series_family(accession: str) -> str:
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError("invalid GEO series accession")
    return re.sub(r"\d{3}$", "nnn", accession)


def parse_series_matrix(payload: bytes, *, accession: str, source_url: str) -> GEOSeriesMatrix:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    metadata: dict[str, list[str]] = {}
    feature_ids: list[str] = []
    values: list[tuple[float | None, ...]] = []
    in_table = False
    reader = csv.reader(io.StringIO(payload.decode("utf-8", errors="replace")), delimiter="\t")
    for row in reader:
        if not row:
            continue
        if row[0] == "!series_matrix_table_begin":
            in_table = True
            continue
        if row[0] == "!series_matrix_table_end":
            break
        if not in_table and row[0].startswith("!Sample_"):
            metadata.setdefault(row[0], []).extend(item.strip('"') for item in row[1:])
            continue
        if in_table:
            if row[0].strip('"') == "ID_REF":
                continue
            feature_ids.append(row[0].strip('"'))
            parsed: list[float | None] = []
            for item in row[1:]:
                try:
                    parsed.append(float(item.strip('"')))
                except ValueError:
                    parsed.append(None)
            values.append(tuple(parsed))
    sample_ids = tuple(metadata.get("!Sample_geo_accession", ()))
    if not sample_ids or not feature_ids or any(len(row) != len(sample_ids) for row in values):
        raise ValueError("GEO Series Matrix is missing an aligned sample/feature table")
    platforms = tuple(metadata.get("!Sample_platform_id", ()))
    if len(set(platforms)) != 1:
        raise ValueError("Series Matrix must contain exactly one platform")
    characteristic_rows = metadata.get("!Sample_characteristics_ch1", [])
    characteristics = tuple(tuple(characteristic_rows[index::len(sample_ids)]) for index in range(len(sample_ids))) if characteristic_rows else tuple(() for _ in sample_ids)
    return GEOSeriesMatrix(
        accession=accession, platform=platforms[0], sample_ids=sample_ids,
        sample_titles=tuple(metadata.get("!Sample_title", ("",) * len(sample_ids))),
        source_names=tuple(metadata.get("!Sample_source_name_ch1", ("",) * len(sample_ids))),
        characteristics=characteristics, feature_ids=tuple(feature_ids), values=tuple(values), source_url=source_url,
    )


def _metadata_value(lines: tuple[str, ...], names: tuple[str, ...]) -> str | None:
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() in names:
            return value.strip() or None
    return None


def parse_phenotypes(matrix: GEOSeriesMatrix, rules: PhenotypeRules) -> tuple[GEOSamplePhenotype, ...]:
    phenotypes = []
    for index, sample_id in enumerate(matrix.sample_ids):
        raw = matrix.characteristics[index]
        combined = " ".join((matrix.sample_titles[index], matrix.source_names[index], *raw)).casefold()
        tumor = any(re.search(pattern, combined, re.I) for pattern in rules.tumor_patterns)
        normal = any(re.search(pattern, combined, re.I) for pattern in rules.normal_patterns)
        label = GEOTumorNormal.UNKNOWN if tumor == normal else GEOTumorNormal.TUMOR if tumor else GEOTumorNormal.NORMAL
        survival_time = _metadata_value(raw, ("survival time", "overall survival", "os time"))
        survival_event = _metadata_value(raw, ("survival event", "vital status", "os event"))
        try:
            parsed_time = float(survival_time) if survival_time else None
        except ValueError:
            parsed_time = None
        parsed_event = None
        if survival_event:
            normalized = survival_event.casefold()
            parsed_event = 1 if normalized in {"1", "dead", "deceased"} else 0 if normalized in {"0", "alive", "living"} else None
        phenotypes.append(GEOSamplePhenotype(
            sample_id=sample_id, disease=_metadata_value(raw, ("disease", "diagnosis")),
            subtype=_metadata_value(raw, ("subtype", "histology")), tissue=matrix.source_names[index] or None,
            tumor_normal=label, survival_time=parsed_time, survival_event=parsed_event,
            treatment=_metadata_value(raw, ("treatment",)), platform=matrix.platform, raw_metadata=raw,
        ))
    return tuple(phenotypes)


def parse_platform_annotation(payload: bytes, *, platform: str, source_url: str) -> GEOPlatformMapping:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    lines = payload.decode("utf-8", errors="replace").splitlines()
    table = [line for line in lines if line and not line.startswith("#") and not line.startswith("!")]
    if not table:
        raise ValueError("GPL annotation table is empty")
    reader = csv.DictReader(io.StringIO("\n".join(table)), delimiter="\t")
    fields = {name.casefold(): name for name in (reader.fieldnames or [])}
    id_field = fields.get("id") or fields.get("id_ref")
    symbol_field = next((fields[name] for name in ("gene symbol", "gene_symbol", "symbol") if name in fields), None)
    if not id_field or not symbol_field:
        raise ValueError("GPL annotation lacks ID and gene-symbol columns")
    mapping: dict[str, tuple[str, ...]] = {}
    ambiguous = 0
    for row in reader:
        identifier = (row.get(id_field) or "").strip()
        symbols = tuple(dict.fromkeys(item.strip() for item in re.split(r"///|//|;|,", row.get(symbol_field) or "") if item.strip() and item.strip() != "---"))
        if identifier and symbols:
            mapping[identifier] = symbols
            ambiguous += len(symbols) > 1
    return GEOPlatformMapping(platform=platform, feature_to_symbols=mapping, source_url=source_url,
                              total_features=len(table) - 1, mapped_features=len(mapping), ambiguous_features=ambiguous)


def assess_geo_compatibility(matrix: GEOSeriesMatrix, phenotypes: tuple[GEOSamplePhenotype, ...],
                             mapping: GEOPlatformMapping, *, candidates: tuple[str, ...],
                             cancer_terms: tuple[str, ...], tissue_terms: tuple[str, ...],
                             min_group_size: int = 3) -> GEOCompatibilityResult:
    tumor = sum(item.tumor_normal == GEOTumorNormal.TUMOR for item in phenotypes)
    normal = sum(item.tumor_normal == GEOTumorNormal.NORMAL for item in phenotypes)
    known = tumor + normal
    quality = known / len(phenotypes) if phenotypes else 0
    unambiguous_symbols = {symbols[0].casefold() for symbols in mapping.feature_to_symbols.values() if len(symbols) == 1}
    coverage = tuple(candidate for candidate in candidates if candidate.casefold() in unambiguous_symbols)
    text = " ".join(item for phenotype in phenotypes for item in (phenotype.disease or "", phenotype.tissue or "")).casefold()
    reasons: list[str] = []
    if not any(term.casefold() in text for term in cancer_terms):
        reasons.append("cancer/subtype compatibility was not established from structured sample metadata")
    if tissue_terms and not any(term.casefold() in text for term in tissue_terms):
        reasons.append("tissue compatibility was not established")
    if tumor < min_group_size or normal < min_group_size:
        reasons.append(f"requires at least {min_group_size} tumor and {min_group_size} normal samples")
    if quality < 0.8:
        reasons.append("fewer than 80% of samples received an unambiguous deterministic phenotype")
    if not coverage:
        reasons.append("none of the requested candidates had an unambiguous platform mapping")
    status = GEOIntegrationStatus.ELIGIBLE if not reasons else GEOIntegrationStatus.NOT_ELIGIBLE
    return GEOCompatibilityResult(accession=matrix.accession, status=status, reasons=tuple(reasons),
                                  tumor_count=tumor, normal_count=normal, candidate_coverage=coverage,
                                  phenotype_quality=quality,
                                  survival_available=any(item.survival_time is not None and item.survival_event is not None for item in phenotypes))


class GEOExpressionClient:
    def __init__(self, *, transport: GEORetrievalTransport | None = None, timeout: float = 120.0,
                 registry: CuratedGEORegistry | None = None) -> None:
        self.transport = transport or UrllibGEORetrievalTransport()
        self.timeout = timeout
        self.registry = registry or load_curated_geo_registry()

    def retrieve(self, accession: str, *, candidates: tuple[str, ...], tcga_directions: dict[str, float],
                 cancer_terms: tuple[str, ...], tissue_terms: tuple[str, ...], rules: PhenotypeRules | None = None) -> GEOParsedCohort:
        family = geo_series_family(accession)
        directory = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{family}/{accession}/matrix/"
        listing = self.transport.get_bytes(directory, self.timeout).decode("utf-8", errors="replace")
        names = sorted(set(re.findall(r'href="([^"]+_series_matrix\.txt\.gz)"', listing)))
        if not names:
            raise ValueError("INTEGRATION_INCOMPLETE: no standardized GEO Series Matrix was available")
        parsed = [parse_series_matrix(self.transport.get_bytes(directory + name, self.timeout), accession=accession, source_url=directory + name) for name in names]
        matrix = next((item for item in parsed if len(set(item.sample_ids)) == len(item.sample_ids)), parsed[0])
        platform_family = re.sub(r"\d{3}$", "nnn", matrix.platform)
        annotation_url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{platform_family}/{matrix.platform}/annot/{matrix.platform}.annot.gz"
        mapping = parse_platform_annotation(self.transport.get_bytes(annotation_url, self.timeout), platform=matrix.platform, source_url=annotation_url)
        phenotypes = parse_phenotypes(matrix, rules or PhenotypeRules())
        curated = self.registry.verified(accession, matrix.platform)
        if curated is not None and any(item.tumor_normal == GEOTumorNormal.UNKNOWN for item in phenotypes):
            phenotypes = apply_curated_phenotypes(matrix, phenotypes, curated)
        compatibility = assess_geo_compatibility(matrix, phenotypes, mapping, candidates=candidates,
                                                 cancer_terms=cancer_terms, tissue_terms=tissue_terms)
        request = None
        if compatibility.status == GEOIntegrationStatus.ELIGIBLE:
            selected_columns = [index for index, item in enumerate(phenotypes) if item.tumor_normal != GEOTumorNormal.UNKNOWN]
            samples = tuple(GEOExpressionSample(sample_id=phenotypes[index].sample_id, group=phenotypes[index].tumor_normal.value) for index in selected_columns)
            symbol_to_rows: dict[str, list[int]] = {}
            for index, feature_id in enumerate(matrix.feature_ids):
                symbols = mapping.feature_to_symbols.get(feature_id, ())
                if len(symbols) == 1 and symbols[0] in compatibility.candidate_coverage:
                    symbol_to_rows.setdefault(symbols[0], []).append(index)
            features_list = []
            for symbol, indices in symbol_to_rows.items():
                values = []
                complete = True
                for column in selected_columns:
                    observed = [matrix.values[index][column] for index in indices if matrix.values[index][column] is not None]
                    if not observed:
                        complete = False
                        break
                    values.append(sum(observed) / len(observed))
                if complete:
                    features_list.append(GEOExpressionFeature(
                        platform_identifier=";".join(matrix.feature_ids[index] for index in indices),
                        gene_symbol=symbol, values=tuple(values),
                    ))
            features = tuple(features_list)
            if not features:
                raise ValueError("NOT_ELIGIBLE: candidate probes contained missing values after phenotype filtering")
            request = GEOReplicationRequest(accession=accession, platform=matrix.platform, cancer=cancer_terms[0],
                                            samples=samples, features=features, candidate_symbols=compatibility.candidate_coverage,
                                            tcga_directions=tcga_directions)
        return GEOParsedCohort(matrix=matrix, phenotypes=phenotypes, mapping=mapping,
                               compatibility=compatibility, replication_request=request)
