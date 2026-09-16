"""Typed, cacheable PubMed search through the NCBI E-utilities API."""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cancer: str = Field(min_length=1)
    concepts: tuple[str, ...] = ()
    max_results: int = Field(default=100, ge=1, le=100)

    @field_validator("cancer")
    @classmethod
    def validate_cancer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("cancer must contain non-whitespace characters")
        return value

    @field_validator("concepts")
    @classmethod
    def validate_concepts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("concepts cannot contain blank values")
        return cleaned


class PubMedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pmid: str = Field(pattern=r"^\d+$")
    title: str = Field(min_length=1)
    abstract: str | None = None
    journal: str | None = None
    publication_year: int | None = Field(default=None, ge=1800, le=2200)
    authors: tuple[str, ...] = ()
    query: str = Field(min_length=1)
    retrieval_timestamp: datetime


class PubMedSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    records: tuple[PubMedRecord, ...] = ()
    total_found: int = Field(ge=0)
    retrieved_at: datetime | None = None
    from_cache: bool = False


class HTTPTransport(Protocol):
    def get(self, url: str, params: dict[str, str], timeout: float) -> bytes: ...


class UrllibTransport:
    """Small standard-library HTTP transport that can be replaced in tests."""

    def get(self, url: str, params: dict[str, str], timeout: float) -> bytes:
        request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "OncoLncAI/0.1"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()


def build_pubmed_query(request: PubMedSearchRequest) -> str:
    """Build a bounded Title/Abstract query from validated concepts."""

    def clause(value: str) -> str:
        escaped = value.replace('"', "")
        return f'"{escaped}"[Title/Abstract]'

    cancer_clause = clause(request.cancer)
    if not request.concepts:
        return cancer_clause
    concepts = " OR ".join(clause(concept) for concept in dict.fromkeys(request.concepts))
    return f"({cancer_clause}) AND ({concepts})"


class PubMedClient:
    """Search PubMed with bounded retries, normalization, and disk caching."""

    def __init__(
        self,
        *,
        transport: HTTPTransport | None = None,
        api_key: str | None = None,
        cache_dir: Path | str | None = Path(".cache/oncolncai/pubmed"),
        timeout: float = 20.0,
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
        self._transport = transport or UrllibTransport()
        # The credential is used only for NCBI requests. It is deliberately kept
        # out of request models, cache keys, provenance, and serialized results.
        self._api_key = api_key.strip() if api_key else None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._cache_ttl = cache_ttl
        self._sleep = sleep
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def search(
        self, request: PubMedSearchRequest
    ) -> ToolResult[PubMedSearchResult]:
        query = build_pubmed_query(request)
        cached = self._read_cache(request, query)
        if cached is not None:
            return self._wrap(cached, query)

        try:
            search_payload = self._request(
                f"{EUTILS_BASE_URL}/esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": str(request.max_results),
                },
            )
            search_data = json.loads(search_payload)
            search_result = search_data["esearchresult"]
            raw_ids = search_result.get("idlist", [])
            pmids = tuple(dict.fromkeys(str(pmid) for pmid in raw_ids if str(pmid).isdigit()))
            total_found = int(search_result.get("count", len(pmids)))

            retrieved_at = self._clock()
            records: tuple[PubMedRecord, ...] = ()
            if pmids:
                fetch_payload = self._request(
                    f"{EUTILS_BASE_URL}/efetch.fcgi",
                    {
                        "db": "pubmed",
                        "id": ",".join(pmids),
                        "retmode": "xml",
                    },
                )
                records = parse_pubmed_xml(fetch_payload, query, retrieved_at)

            result = PubMedSearchResult(
                query=query,
                records=records,
                total_found=total_found,
                retrieved_at=retrieved_at,
            )
            self._write_cache(request, result)
            return self._wrap(result, query)
        except (HTTPError, URLError, ET.ParseError, KeyError, TypeError, ValueError) as exc:
            return ToolResult[PubMedSearchResult](
                status=ResultStatus.FAILURE,
                error=ToolError(
                    code="pubmed_request_failed",
                    message=str(exc) or type(exc).__name__,
                    retryable=(
                        (isinstance(exc, HTTPError) and (exc.code == 429 or exc.code >= 500))
                        or (isinstance(exc, URLError) and not isinstance(exc, HTTPError))
                    ),
                ),
                provenance=[Provenance(source="NCBI PubMed E-utilities", query=query)],
            )

    def _request(self, url: str, params: dict[str, str]) -> bytes:
        request_params = dict(params)
        if self._api_key:
            request_params["api_key"] = self._api_key
        for attempt in range(self._max_retries + 1):
            try:
                return self._transport.get(url, request_params, self._timeout)
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self._max_retries:
                    raise
            except URLError:
                if attempt == self._max_retries:
                    raise
            self._sleep(self._backoff_seconds * (2**attempt))
        raise RuntimeError("unreachable retry state")

    def _cache_path(self, request: PubMedSearchRequest) -> Path | None:
        if self._cache_dir is None:
            return None
        cache_key = hashlib.sha256(
            request.model_dump_json().encode("utf-8")
        ).hexdigest()
        return self._cache_dir / f"{cache_key}.json"

    def _read_cache(
        self, request: PubMedSearchRequest, query: str
    ) -> PubMedSearchResult | None:
        path = self._cache_path(request)
        if path is None or not path.exists():
            return None
        try:
            cached = PubMedSearchResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if cached.query != query:
            return None
        if cached.retrieved_at is None:
            return None
        if self._cache_ttl is not None and self._clock() - cached.retrieved_at > self._cache_ttl:
            return None
        return PubMedSearchResult.model_validate(
            {**cached.model_dump(), "from_cache": True}
        )

    def _write_cache(
        self, request: PubMedSearchRequest, result: PubMedSearchResult
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
                temporary.write(result.model_dump_json(indent=2))
                temporary.write("\n")
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _wrap(
        result: PubMedSearchResult, query: str
    ) -> ToolResult[PubMedSearchResult]:
        provenance = [Provenance(source="NCBI PubMed E-utilities", query=query)]
        provenance.extend(
            Provenance(
                source="NCBI PubMed",
                source_id=record.pmid,
                query=query,
            )
            for record in result.records
        )
        return ToolResult[PubMedSearchResult](
            status=ResultStatus.SUCCESS,
            data=result,
            provenance=provenance,
        )


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return " ".join(value.split()) or None


def _publication_year(article: ET.Element) -> int | None:
    for path in (
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//DateCompleted/Year",
    ):
        value = _text(article.find(path))
        if value and value.isdigit():
            return int(value)
    medline_date = _text(article.find(".//JournalIssue/PubDate/MedlineDate"))
    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|21\d{2})\b", medline_date or "")
    return int(match.group(1)) if match else None


def parse_pubmed_xml(
    payload: bytes, query: str, retrieval_timestamp: datetime
) -> tuple[PubMedRecord, ...]:
    """Normalize PubMed XML records and discard duplicate PMIDs."""

    root = ET.fromstring(payload)
    records: list[PubMedRecord] = []
    seen_pmids: set[str] = set()
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//Article/ArticleTitle"))
        if not pmid or not pmid.isdigit() or pmid in seen_pmids or not title:
            continue

        abstract_parts: list[str] = []
        for abstract_element in article.findall(".//Article/Abstract/AbstractText"):
            text = _text(abstract_element)
            if text:
                label = abstract_element.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)

        authors: list[str] = []
        for author in article.findall(".//Article/AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            last_name = _text(author.find("LastName"))
            initials = _text(author.find("Initials"))
            name = collective or " ".join(value for value in (last_name, initials) if value)
            if name:
                authors.append(name)

        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=" ".join(abstract_parts) or None,
                journal=_text(article.find(".//Article/Journal/Title")),
                publication_year=_publication_year(article),
                authors=tuple(authors),
                query=query,
                retrieval_timestamp=retrieval_timestamp,
            )
        )
        seen_pmids.add(pmid)
    return tuple(records)


def search_pubmed(
    request: PubMedSearchRequest,
    *,
    client: PubMedClient | None = None,
) -> ToolResult[PubMedSearchResult]:
    """Search PubMed directly, with dependency injection available for tests."""

    return (client or PubMedClient()).search(request)
