"""Deterministic discovery of TCGA analysis capabilities through GDC."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


GDC_API_BASE_URL = "https://api.gdc.cancer.gov"
MIN_DE_TUMOR_SAMPLES = 20
MIN_DE_NORMAL_SAMPLES = 10
MIN_SURVIVAL_SAMPLES = 20
MAX_DISCOVERY_RECORDS = 10_000


class TCCapabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")


class DataCapabilities(BaseModel):
    """Structured observations that a planner can use without raw GDC payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(pattern=r"^TCGA-[A-Z0-9]+$")
    rna_seq_available: bool
    clinical_available: bool
    survival_available: bool
    tumor_samples: int = Field(ge=0)
    normal_samples: int = Field(ge=0)
    survival_samples: int = Field(ge=0)
    de_recommended: bool
    survival_recommended: bool
    discovery_complete: bool
    reasons: tuple[str, ...]
    checked_at: datetime
    from_cache: bool = False

    @model_validator(mode="after")
    def validate_recommendations(self) -> DataCapabilities:
        if self.de_recommended and (
            not self.rna_seq_available
            or not self.discovery_complete
            or self.tumor_samples < MIN_DE_TUMOR_SAMPLES
            or self.normal_samples < MIN_DE_NORMAL_SAMPLES
        ):
            raise ValueError("DE recommendation is inconsistent with discovered capabilities")
        if self.survival_recommended and (
            not self.survival_available
            or not self.discovery_complete
            or self.survival_samples < MIN_SURVIVAL_SAMPLES
        ):
            raise ValueError("survival recommendation is inconsistent with capabilities")
        return self


class GDCTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class UrllibGDCTransport:
    """Standard-library JSON transport, replaceable by a fake in tests."""

    def post_json(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "OncoLncAI/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())


class GDCCapabilityClient:
    """Inspect whether a TCGA project supports DE and survival analysis."""

    def __init__(
        self,
        *,
        transport: GDCTransport | None = None,
        cache_dir: Path | None = Path(".cache/oncolncai/gdc_capabilities"),
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        cache_ttl: timedelta | None = timedelta(hours=24),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        if cache_ttl is not None and cache_ttl.total_seconds() < 0:
            raise ValueError("cache_ttl cannot be negative")
        self._transport = transport or UrllibGDCTransport()
        self._cache_dir = cache_dir
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._cache_ttl = cache_ttl
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def inspect(
        self, request: TCCapabilityRequest
    ) -> ToolResult[DataCapabilities]:
        cached = self._read_cache(request)
        if cached is not None:
            return self._wrap(cached)

        try:
            file_response = self._request(
                f"{GDC_API_BASE_URL}/files",
                _rna_discovery_payload(request.project_id),
            )
            clinical_response = self._request(
                f"{GDC_API_BASE_URL}/cases",
                _clinical_discovery_payload(request.project_id),
            )
            capabilities = _summarize_capabilities(
                request.project_id,
                file_response,
                clinical_response,
                checked_at=self._clock(),
            )
            self._write_cache(request, capabilities)
            return self._wrap(capabilities)
        except (HTTPError, URLError, KeyError, TypeError, ValueError) as exc:
            return ToolResult[DataCapabilities](
                status=ResultStatus.FAILURE,
                error=ToolError(
                    code="gdc_capability_request_failed",
                    message=str(exc) or type(exc).__name__,
                    retryable=(
                        (isinstance(exc, HTTPError) and (exc.code == 429 or exc.code >= 500))
                        or (isinstance(exc, URLError) and not isinstance(exc, HTTPError))
                    ),
                ),
                provenance=[
                    Provenance(source="NCI GDC API", source_id=request.project_id)
                ],
            )

    def _request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            try:
                return self._transport.post_json(url, payload, self._timeout)
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self._max_retries:
                    raise
            except URLError:
                if attempt == self._max_retries:
                    raise
            self._sleep(self._backoff_seconds * (2**attempt))
        raise RuntimeError("unreachable retry state")

    def _cache_path(self, request: TCCapabilityRequest) -> Path | None:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, request: TCCapabilityRequest) -> DataCapabilities | None:
        path = self._cache_path(request)
        if path is None or not path.exists():
            return None
        try:
            cached = DataCapabilities.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if cached.project_id != request.project_id:
            return None
        if self._cache_ttl is not None and self._clock() - cached.checked_at > self._cache_ttl:
            return None
        return DataCapabilities.model_validate({**cached.model_dump(), "from_cache": True})

    def _write_cache(
        self, request: TCCapabilityRequest, capabilities: DataCapabilities
    ) -> None:
        path = self._cache_path(request)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(capabilities.model_dump_json(indent=2))
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _wrap(capabilities: DataCapabilities) -> ToolResult[DataCapabilities]:
        warnings = []
        if not capabilities.discovery_complete:
            warnings.append("GDC discovery response was truncated; recommendations are disabled.")
        return ToolResult[DataCapabilities](
            status=ResultStatus.SUCCESS,
            data=capabilities,
            warnings=warnings,
            provenance=[
                Provenance(
                    source="NCI GDC API",
                    source_id=capabilities.project_id,
                    dataset_version=capabilities.checked_at.isoformat(),
                )
            ],
        )


def _project_filter(project_id: str) -> dict[str, Any]:
    return {
        "op": "in",
        "content": {"field": "project.project_id", "value": [project_id]},
    }


def _rna_discovery_payload(project_id: str) -> dict[str, Any]:
    return {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [project_id],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": ["Gene Expression Quantification"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "analysis.workflow_type",
                        "value": ["STAR - Counts"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": ["Primary Tumor", "Solid Tissue Normal"],
                    },
                },
            ],
        },
        "fields": "file_id,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": MAX_DISCOVERY_RECORDS,
    }


def _clinical_discovery_payload(project_id: str) -> dict[str, Any]:
    return {
        "filters": _project_filter(project_id),
        "fields": (
            "case_id,demographic.vital_status,demographic.days_to_death,"
            "diagnoses.days_to_death,diagnoses.days_to_last_follow_up,"
            "follow_ups.days_to_follow_up"
        ),
        "format": "JSON",
        "size": MAX_DISCOVERY_RECORDS,
    }


def _pagination(response: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    data = response["data"]
    hits = data["hits"]
    if not isinstance(hits, list):
        raise TypeError("GDC response data.hits must be a list")
    total = int(data["pagination"]["total"])
    return hits, total


def _sample_counts(file_hits: list[dict[str, Any]]) -> tuple[int, int]:
    tumor_ids: set[str] = set()
    normal_ids: set[str] = set()
    for file_record in file_hits:
        for case in file_record.get("cases", []):
            for sample in case.get("samples", []):
                sample_id = sample.get("submitter_id")
                sample_type = sample.get("sample_type")
                if not sample_id:
                    continue
                if sample_type == "Primary Tumor":
                    tumor_ids.add(str(sample_id))
                elif sample_type == "Solid Tissue Normal":
                    normal_ids.add(str(sample_id))
    return len(tumor_ids), len(normal_ids)


def _values_for_key(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                values.append(current_value)
            values.extend(_values_for_key(current_value, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_values_for_key(item, key))
    return values


def _valid_day(values: list[Any]) -> bool:
    for value in values:
        try:
            if value is not None and float(value) >= 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _survival_sample_count(case_hits: list[dict[str, Any]]) -> int:
    usable = 0
    for case in case_hits:
        statuses = {str(value).casefold() for value in _values_for_key(case, "vital_status")}
        has_death_time = _valid_day(_values_for_key(case, "days_to_death"))
        has_follow_up = _valid_day(
            _values_for_key(case, "days_to_last_follow_up")
            + _values_for_key(case, "days_to_follow_up")
        )
        if ("dead" in statuses and has_death_time) or (
            "alive" in statuses and has_follow_up
        ):
            usable += 1
    return usable


def _summarize_capabilities(
    project_id: str,
    file_response: dict[str, Any],
    clinical_response: dict[str, Any],
    *,
    checked_at: datetime,
) -> DataCapabilities:
    file_hits, file_total = _pagination(file_response)
    case_hits, case_total = _pagination(clinical_response)
    tumor_samples, normal_samples = _sample_counts(file_hits)
    survival_samples = _survival_sample_count(case_hits)
    discovery_complete = len(file_hits) >= file_total and len(case_hits) >= case_total
    rna_available = tumor_samples + normal_samples > 0
    clinical_available = case_total > 0
    survival_available = survival_samples > 0

    de_recommended = (
        discovery_complete
        and rna_available
        and tumor_samples >= MIN_DE_TUMOR_SAMPLES
        and normal_samples >= MIN_DE_NORMAL_SAMPLES
    )
    survival_recommended = (
        discovery_complete
        and survival_available
        and survival_samples >= MIN_SURVIVAL_SAMPLES
    )

    reasons: list[str] = []
    if not discovery_complete:
        reasons.append("GDC response was truncated; sample counts are incomplete.")
    if not rna_available:
        reasons.append("No STAR-count RNA-seq samples were discovered.")
    elif not discovery_complete:
        reasons.append("DE not recommended: RNA-seq discovery is incomplete.")
    elif tumor_samples < MIN_DE_TUMOR_SAMPLES:
        reasons.append(
            f"DE not recommended: {tumor_samples} primary tumors; minimum is {MIN_DE_TUMOR_SAMPLES}."
        )
    elif normal_samples < MIN_DE_NORMAL_SAMPLES:
        reasons.append(
            f"DE not recommended: {normal_samples} solid-tissue normals; minimum is {MIN_DE_NORMAL_SAMPLES}."
        )
    else:
        reasons.append("DE recommended: minimum tumor and normal sample counts are met.")

    if not clinical_available:
        reasons.append("No clinical cases were discovered.")
    elif not survival_available:
        reasons.append("No cases with usable vital status and follow-up time were discovered.")
    elif not discovery_complete:
        reasons.append("Survival not recommended: clinical discovery is incomplete.")
    elif survival_samples < MIN_SURVIVAL_SAMPLES:
        reasons.append(
            f"Survival not recommended: {survival_samples} usable cases; minimum is {MIN_SURVIVAL_SAMPLES}."
        )
    else:
        reasons.append("Survival recommended: minimum usable case count is met.")

    return DataCapabilities(
        project_id=project_id,
        rna_seq_available=rna_available,
        clinical_available=clinical_available,
        survival_available=survival_available,
        tumor_samples=tumor_samples,
        normal_samples=normal_samples,
        survival_samples=survival_samples,
        de_recommended=de_recommended,
        survival_recommended=survival_recommended,
        discovery_complete=discovery_complete,
        reasons=tuple(reasons),
        checked_at=checked_at,
    )


def inspect_tcga_capabilities(
    request: TCCapabilityRequest,
    *,
    client: GDCCapabilityClient | None = None,
) -> ToolResult[DataCapabilities]:
    """Inspect a TCGA project's capabilities through the supplied GDC client."""

    return (client or GDCCapabilityClient()).inspect(request)
