"""Targeted native-GDC raw STAR-count retrieval and matrix assembly."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from oncolncai.differential_expression import (
    CountFeature, DifferentialExpressionMethod, DifferentialExpressionRequest,
    ExpressionDataType, ExpressionGroup, ExpressionSample,
)
from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult
from oncolncai.tcga_discovery import LNCRNA_BIOTYPES, normalize_ensembl_id


GDC_API = "https://api.gdc.cancer.gov"


class ExpressionPurpose(str):
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    SURVIVAL = "survival"


class ExpressionSourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    purpose: str
    upstream_source: str
    distribution_source: str
    representation: str
    transform: str | None = None
    dataset_identifier: str


class GDCCountFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    file_id: str
    file_name: str
    md5sum: str = Field(pattern=r"^[0-9a-f]{32}$")
    file_size: int = Field(gt=0)
    sample_id: str
    case_id: str
    sample_type: str
    group: ExpressionGroup


class GDCRawCountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    max_tumor_samples: int | None = Field(default=100, ge=20)
    max_normal_samples: int | None = Field(default=100, ge=10)
    min_tumor_samples: int = Field(default=20, ge=2)
    min_normal_samples: int = Field(default=10, ge=2)


class RawCountPreparationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    queried_files: int
    selected_files: int
    tumor_samples: int
    normal_samples: int
    genes_in_source: int
    lncrnas_retained: int
    cache_reused: bool


class PreparedGDCRawCountCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str
    samples: tuple[GDCCountFile, ...]
    features: tuple[CountFeature, ...]
    preparation: RawCountPreparationSummary
    source: ExpressionSourceMetadata
    query: dict
    file_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    matrix_cache_path: str
    retrieved_at: datetime

    def de_request(self) -> DifferentialExpressionRequest:
        return DifferentialExpressionRequest(
            project_id=self.project_id,
            samples=tuple(ExpressionSample(sample_id=item.sample_id, group=item.group) for item in self.samples),
            features=self.features,
            method=DifferentialExpressionMethod.AUTO,
            input_data_type=ExpressionDataType.RAW_INTEGER_COUNTS,
            expression_transform="none",
            expression_source_dataset=self.source.dataset_identifier,
            upstream_source=self.source.upstream_source,
            distribution_source=self.source.distribution_source,
            count_source=f"Native GDC STAR - Counts matrix sha256:{self.matrix_checksum}",
            annotation_source="GDC STAR count file gene_id/gene_name/gene_type columns",
        )


class GDCCountTransport(Protocol):
    def get_json(self, url: str, timeout: float) -> dict: ...
    def download(self, url: str, destination: Path, timeout: float) -> None: ...


class UrllibGDCCountTransport:
    def get_json(self, url: str, timeout: float) -> dict:
        error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "OncoLncAI/0.1"}), timeout=timeout) as response:  # noqa: S310
                    return json.load(response)
            except Exception as exc:
                error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise OSError(f"GDC metadata request failed after 3 attempts: {error}") from error

    def download(self, url: str, destination: Path, timeout: float) -> None:
        error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "OncoLncAI/0.1"}), timeout=timeout) as response, destination.open("wb") as output:  # noqa: S310
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                return
            except Exception as exc:
                error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise OSError(f"GDC file download failed after 3 attempts: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to validate the GDC-provided checksum
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _query_payload(project_id: str) -> dict:
    return {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]}


def parse_gdc_file_records(payload: dict) -> tuple[GDCCountFile, ...]:
    records: list[GDCCountFile] = []
    for hit in payload.get("data", {}).get("hits", []):
        cases = hit.get("cases") or []
        if len(cases) != 1:
            continue
        samples = cases[0].get("samples") or []
        eligible = [sample for sample in samples if sample.get("sample_type") in {"Primary Tumor", "Solid Tissue Normal"}]
        if len(eligible) != 1:
            continue
        sample = eligible[0]
        group = ExpressionGroup.CASE if sample["sample_type"] == "Primary Tumor" else ExpressionGroup.CONTROL
        records.append(GDCCountFile(
            file_id=hit.get("file_id") or hit["id"], file_name=hit["file_name"], md5sum=hit["md5sum"],
            file_size=hit["file_size"], sample_id=sample["submitter_id"], case_id=cases[0]["submitter_id"],
            sample_type=sample["sample_type"], group=group,
        ))
    unique: dict[tuple[str, ExpressionGroup], GDCCountFile] = {}
    for record in sorted(records, key=lambda item: (item.sample_id, item.file_id)):
        unique.setdefault((record.sample_id, record.group), record)
    return tuple(unique.values())


def _select_files(records: tuple[GDCCountFile, ...], request: GDCRawCountRequest) -> tuple[GDCCountFile, ...]:
    tumor = sorted((item for item in records if item.group == ExpressionGroup.CASE), key=lambda item: item.sample_id)
    normal = sorted((item for item in records if item.group == ExpressionGroup.CONTROL), key=lambda item: item.sample_id)
    if request.max_tumor_samples is not None:
        tumor = tumor[:request.max_tumor_samples]
    if request.max_normal_samples is not None:
        normal = normal[:request.max_normal_samples]
    if len(tumor) < request.min_tumor_samples or len(normal) < request.min_normal_samples:
        raise ValueError(f"raw-count cohort is not eligible: tumor={len(tumor)}, normal={len(normal)}")
    return tuple(tumor + normal)


def parse_gdc_star_counts(path: Path) -> dict[str, tuple[str, str, int]]:
    result: dict[str, tuple[str, str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader((line for line in source if not line.startswith("#")), delimiter="\t")
        required = {"gene_id", "gene_name", "gene_type", "unstranded"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("GDC STAR-count file lacks required raw-count columns")
        for row in reader:
            if row["gene_id"].endswith("_PAR_Y"):
                # GENCODE duplicates pseudoautosomal genes on chrY; retain the canonical
                # non-PAR_Y row so version normalization remains one gene per feature.
                continue
            identifier = normalize_ensembl_id(row["gene_id"])
            if not identifier.startswith("ENSG") or row["gene_type"] not in LNCRNA_BIOTYPES:
                continue
            raw = row["unstranded"]
            if not raw.isdigit():
                raise ValueError(f"non-integer raw count for {identifier}")
            if identifier in result:
                raise ValueError(f"duplicate normalized Ensembl ID in GDC file: {identifier}")
            result[identifier] = (row["gene_name"], row["gene_type"], int(raw))
    if not result:
        raise ValueError("GDC file contained no recognized lncRNA raw counts")
    return result


class GDCRawCountClient:
    def __init__(self, *, transport: GDCCountTransport | None = None,
                 cache_dir: Path = Path(".cache/oncolncai/gdc_raw_counts"), timeout: float = 180.0) -> None:
        self.transport = transport or UrllibGDCCountTransport()
        self.cache_dir = cache_dir
        self.timeout = timeout

    def prepare(self, request: GDCRawCountRequest) -> ToolResult[PreparedGDCRawCountCohort]:
        filters = _query_payload(request.project_id)
        fields = "file_id,file_name,file_size,md5sum,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type"
        url = f"{GDC_API}/files?" + urlencode({"filters": json.dumps(filters), "format": "JSON", "size": "10000", "fields": fields})
        try:
            payload = self.transport.get_json(url, self.timeout)
            all_records = parse_gdc_file_records(payload)
            selected = _select_files(all_records, request)
            signature = hashlib.sha256(json.dumps([item.model_dump() for item in selected], sort_keys=True).encode()).hexdigest()
            project_dir = self.cache_dir / request.project_id / signature
            matrix_path = project_dir / "lncrna_raw_counts.tsv.gz"
            manifest_path = project_dir / "manifest.json"
            reused = matrix_path.exists() and manifest_path.exists()
            if not reused:
                project_dir.mkdir(parents=True, exist_ok=True)
                columns: dict[str, list[int]] = {}
                symbols: dict[str, str] = {}
                for record in selected:
                    source_path = self._obtain(record, project_dir / "files")
                    parsed = parse_gdc_star_counts(source_path)
                    if columns and set(parsed) != set(columns):
                        raise ValueError(f"gene set mismatch in GDC file {record.file_id}")
                    if not columns:
                        columns = {identifier: [] for identifier in parsed}
                    for identifier, (symbol, _biotype, count) in parsed.items():
                        columns[identifier].append(count)
                        symbols[identifier] = symbol
                with gzip.open(matrix_path, "wt", encoding="utf-8", newline="") as output:
                    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
                    writer.writerow(["gene_id", "gene_symbol", *(item.sample_id for item in selected)])
                    for identifier in sorted(columns):
                        writer.writerow([identifier, symbols[identifier], *columns[identifier]])
                manifest_path.write_text(json.dumps({"query": filters, "files": [item.model_dump(mode="json") for item in selected]}, indent=2) + "\n", encoding="utf-8")
            features = self._read_matrix(matrix_path, len(selected))
            cohort = PreparedGDCRawCountCohort(
                project_id=request.project_id, samples=selected, features=features,
                preparation=RawCountPreparationSummary(queried_files=len(all_records), selected_files=len(selected),
                    tumor_samples=sum(item.group == ExpressionGroup.CASE for item in selected),
                    normal_samples=sum(item.group == ExpressionGroup.CONTROL for item in selected),
                    genes_in_source=len(features), lncrnas_retained=len(features), cache_reused=reused),
                source=ExpressionSourceMetadata(purpose=ExpressionPurpose.DIFFERENTIAL_EXPRESSION,
                    upstream_source="NCI GDC / TCGA", distribution_source="Native NCI GDC API",
                    representation="raw_integer_counts", transform=None,
                    dataset_identifier=f"{request.project_id} Gene Expression Quantification / STAR - Counts"),
                query=filters, file_manifest_checksum=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                matrix_checksum=_sha256(matrix_path), matrix_cache_path=str(matrix_path), retrieved_at=datetime.now(timezone.utc),
            )
            return ToolResult(status=ResultStatus.SUCCESS, data=cohort, provenance=[
                Provenance(source="NCI GDC API", source_id=request.project_id, query=json.dumps(filters, sort_keys=True), dataset_version=f"manifest-sha256:{cohort.file_manifest_checksum}"),
                Provenance(source="Native GDC raw-count cache", source_id=str(matrix_path), dataset_version=f"sha256:{cohort.matrix_checksum}"),
            ])
        except Exception as exc:
            return ToolResult(status=ResultStatus.FAILURE,
                              error=ToolError(code="gdc_raw_count_preparation_failed", message=str(exc) or type(exc).__name__, retryable=True),
                              provenance=[Provenance(source="NCI GDC API", source_id=request.project_id, query=json.dumps(filters, sort_keys=True))])

    def _obtain(self, record: GDCCountFile, directory: Path) -> Path:
        path = directory / f"{record.file_id}.tsv"
        if path.exists() and path.stat().st_size == record.file_size and _md5(path) == record.md5sum:
            return path
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            self.transport.download(f"{GDC_API}/data/{record.file_id}", temporary_path, self.timeout)
            if temporary_path.stat().st_size != record.file_size or _md5(temporary_path) != record.md5sum:
                raise ValueError(f"GDC checksum/size validation failed for {record.file_id}")
            temporary_path.replace(path)
            return path
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _read_matrix(path: Path, sample_count: int) -> tuple[CountFeature, ...]:
        features = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            reader = csv.reader(source, delimiter="\t")
            header = next(reader)
            if len(header) - 2 != sample_count:
                raise ValueError("cached raw-count matrix sample width mismatch")
            for row in reader:
                counts = tuple(int(value) for value in row[2:])
                if any(value < 0 for value in counts):
                    raise ValueError("cached matrix contains a negative count")
                features.append(CountFeature(feature_id=row[0], gene_symbol=row[1], counts=counts))
        return tuple(features)
