"""Prepare real TCGA lncRNA survival inputs from pinned public matrices."""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult
from oncolncai.survival import SurvivalEndpoint, SurvivalRequest, SurvivalSample


PAN_CANCER_SURVIVAL_URL = (
    "https://pancanatlas.xenahubs.net/download/"
    "Survival_SupplementalTable_S1_20171025_xena_sp"
)


def gdc_expression_url(project_id: str) -> str:
    return f"https://gdc.xenahubs.net/download/{project_id}.star_counts.tsv.gz"


class TCGASurvivalDataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    gene: str = Field(min_length=1)
    ensembl_gene_id: str = Field(pattern=r"^ENSG\d{11}$")
    endpoint: SurvivalEndpoint = SurvivalEndpoint.OS
    expression_url: str | None = None
    clinical_url: str = PAN_CANCER_SURVIVAL_URL
    expression_scale: str = "Xena GDC-hub transformed STAR-count value"


class CohortPreparationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expression_columns: int = Field(ge=0)
    primary_tumor_columns: int = Field(ge=0)
    unique_primary_patients: int = Field(ge=0)
    clinical_project_rows: int = Field(ge=0)
    clinical_patients_with_endpoint: int = Field(ge=0)
    aligned_samples: int = Field(ge=0)
    aligned_events: int = Field(ge=0)
    excluded_non_primary: int = Field(ge=0)
    excluded_duplicate_aliquots: int = Field(ge=0)
    excluded_missing_endpoint: int = Field(ge=0)
    excluded_unmatched_expression: int = Field(ge=0)


class ClinicalCovariates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str
    age: float | None = Field(default=None, gt=0)
    sex: str | None = None
    pathological_stage: str | None = None


class PreparedTCGASurvivalCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    gene: str
    ensembl_gene_id: str
    endpoint: SurvivalEndpoint
    expression_scale: str
    samples: tuple[SurvivalSample, ...]
    clinical_covariates: tuple[ClinicalCovariates, ...] = ()
    preparation: CohortPreparationSummary
    expression_url: str
    clinical_url: str
    expression_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clinical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime

    def survival_request(self) -> SurvivalRequest:
        return SurvivalRequest(
            project_id=self.project_id,
            gene=self.gene,
            samples=self.samples,
            endpoint=self.endpoint,
            cohort_source=f"TCGA PanCanAtlas survival table sha256:{self.clinical_sha256}",
            expression_source=(
                f"UCSC Xena GDC-hub STAR counts sha256:{self.expression_sha256}; "
                f"gene={self.ensembl_gene_id}; scale={self.expression_scale}"
            ),
        )


class DownloadTransport(Protocol):
    def download(self, url: str, destination: Path, timeout: float) -> None: ...


class UrllibDownloadTransport:
    def download(self, url: str, destination: Path, timeout: float) -> None:
        request = Request(url, headers={"User-Agent": "OncoLncAI/0.1"})
        with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:  # noqa: S310
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


class TCGASurvivalDataClient:
    """Download, checksum, filter, and align public TCGA survival inputs."""

    def __init__(
        self,
        *,
        transport: DownloadTransport | None = None,
        cache_dir: Path = Path(".cache/oncolncai/tcga_survival"),
        timeout: float = 120.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._transport = transport or UrllibDownloadTransport()
        self._cache_dir = cache_dir
        self._timeout = timeout
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def prepare(
        self, request: TCGASurvivalDataRequest
    ) -> ToolResult[PreparedTCGASurvivalCohort]:
        expression_url = request.expression_url or gdc_expression_url(request.project_id)
        try:
            expression_path = self._obtain(expression_url)
            clinical_path = self._obtain(request.clinical_url)
            cohort = prepare_tcga_survival_cohort(
                request,
                expression_path=expression_path,
                clinical_path=clinical_path,
                expression_url=expression_url,
                retrieved_at=self._clock(),
            )
            return ToolResult[PreparedTCGASurvivalCohort](
                status=ResultStatus.SUCCESS,
                data=cohort,
                provenance=_cohort_provenance(cohort),
            )
        except (HTTPError, URLError, OSError, UnicodeError, ValueError, KeyError) as exc:
            return ToolResult[PreparedTCGASurvivalCohort](
                status=ResultStatus.FAILURE,
                error=ToolError(
                    code="tcga_survival_preparation_failed",
                    message=str(exc) or type(exc).__name__,
                    retryable=isinstance(exc, (HTTPError, URLError)),
                ),
                provenance=[
                    Provenance(source="UCSC Xena GDC Hub", source_id=expression_url),
                    Provenance(source="TCGA PanCanAtlas", source_id=request.clinical_url),
                ],
            )

    def _obtain(self, url: str) -> Path:
        suffix = ".tsv.gz" if url.endswith(".gz") else ".tsv"
        path = self._cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}{suffix}"
        if path.exists() and path.stat().st_size:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            self._transport.download(url, temporary_path, self._timeout)
            if temporary_path.stat().st_size == 0:
                raise ValueError(f"downloaded source is empty: {url}")
            temporary_path.replace(path)
            return path
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _patient_id(sample_id: str) -> str:
    parts = sample_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"invalid TCGA sample barcode: {sample_id}")
    return "-".join(parts[:3])


def _is_primary_tumor(sample_id: str) -> bool:
    parts = sample_id.split("-")
    return len(parts) >= 4 and parts[3][:2] == "01"


def _read_expression(
    path: Path, ensembl_gene_id: str
) -> tuple[dict[str, float], int, int, int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        values: list[str] | None = None
        for row in reader:
            if row and row[0].split(".", 1)[0] == ensembl_gene_id:
                values = row[1:]
                break
    if values is None:
        raise ValueError(f"gene {ensembl_gene_id} was not found in expression matrix")
    if len(values) != len(sample_ids):
        raise ValueError("expression row width does not match matrix header")
    primary = [(sample, value) for sample, value in zip(sample_ids, values) if _is_primary_tumor(sample)]
    selected: dict[str, tuple[str, float]] = {}
    for sample, raw_value in primary:
        value = float(raw_value)
        patient = _patient_id(sample)
        current = selected.get(patient)
        if current is None or sample < current[0]:
            selected[patient] = (sample, value)
    return (
        {patient: value for patient, (_, value) in selected.items()},
        len(sample_ids),
        len(primary),
        len(primary) - len(selected),
    )


def _read_clinical(
    path: Path, project_id: str, endpoint: SurvivalEndpoint
) -> tuple[dict[str, tuple[float, int]], dict[str, ClinicalCovariates], int, int]:
    project = project_id.removeprefix("TCGA-")
    time_field = f"{endpoint.value}.time"
    event_field = endpoint.value
    rows = 0
    missing = 0
    endpoints: dict[str, tuple[float, int]] = {}
    covariates: dict[str, ClinicalCovariates] = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if row.get("cancer type abbreviation") != project:
                continue
            rows += 1
            raw_time = row.get(time_field, "").strip()
            raw_event = row.get(event_field, "").strip()
            try:
                time = float(raw_time)
                event = int(raw_event)
            except ValueError:
                missing += 1
                continue
            if time <= 0 or event not in (0, 1):
                missing += 1
                continue
            patient = row.get("_PATIENT", "").strip() or _patient_id(row["sample"])
            previous = endpoints.get(patient)
            if previous is not None and previous != (time, event):
                raise ValueError(f"conflicting endpoint rows for patient {patient}")
            endpoints[patient] = (time, event)
            raw_age = row.get("age_at_initial_pathologic_diagnosis", "").strip()
            try:
                age = float(raw_age)
            except ValueError:
                age = None
            covariates[patient] = ClinicalCovariates(
                patient_id=patient,
                age=age if age and age > 0 else None,
                sex=row.get("gender", "").strip() or None,
                pathological_stage=(row.get("ajcc_pathologic_tumor_stage", "").strip()
                                    or row.get("clinical_stage", "").strip() or None),
            )
    if rows == 0:
        raise ValueError(f"no clinical rows found for {project_id}")
    return endpoints, covariates, rows, missing


def prepare_tcga_survival_cohort(
    request: TCGASurvivalDataRequest,
    *,
    expression_path: Path,
    clinical_path: Path,
    expression_url: str | None = None,
    retrieved_at: datetime | None = None,
) -> PreparedTCGASurvivalCohort:
    expression, columns, primary_columns, duplicate_aliquots = _read_expression(
        expression_path, request.ensembl_gene_id
    )
    endpoints, covariates, clinical_rows, missing_endpoint = _read_clinical(
        clinical_path, request.project_id, request.endpoint
    )
    samples = tuple(
        SurvivalSample(sample_id=patient, time=endpoints[patient][0], event=endpoints[patient][1], expression=value)
        for patient, value in sorted(expression.items())
        if patient in endpoints
    )
    if not samples:
        raise ValueError("no expression and survival samples could be aligned")
    cohort = PreparedTCGASurvivalCohort(
        project_id=request.project_id,
        gene=request.gene,
        ensembl_gene_id=request.ensembl_gene_id,
        endpoint=request.endpoint,
        expression_scale=request.expression_scale,
        samples=samples,
        clinical_covariates=tuple(covariates[patient] for patient in sorted(expression) if patient in endpoints),
        preparation=CohortPreparationSummary(
            expression_columns=columns,
            primary_tumor_columns=primary_columns,
            unique_primary_patients=len(expression),
            clinical_project_rows=clinical_rows,
            clinical_patients_with_endpoint=len(endpoints),
            aligned_samples=len(samples),
            aligned_events=sum(sample.event for sample in samples),
            excluded_non_primary=columns - primary_columns,
            excluded_duplicate_aliquots=duplicate_aliquots,
            excluded_missing_endpoint=missing_endpoint,
            excluded_unmatched_expression=len(expression) - len(samples),
        ),
        expression_url=expression_url or request.expression_url or gdc_expression_url(request.project_id),
        clinical_url=request.clinical_url,
        expression_sha256=_sha256(expression_path),
        clinical_sha256=_sha256(clinical_path),
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
    )
    return cohort


def _cohort_provenance(cohort: PreparedTCGASurvivalCohort) -> list[Provenance]:
    return [
        Provenance(
            source="UCSC Xena GDC Hub",
            source_id=cohort.expression_url,
            dataset_version=f"sha256:{cohort.expression_sha256}",
        ),
        Provenance(
            source="TCGA PanCanAtlas curated survival endpoints",
            source_id=cohort.clinical_url,
            dataset_version=f"sha256:{cohort.clinical_sha256}",
        ),
    ]
