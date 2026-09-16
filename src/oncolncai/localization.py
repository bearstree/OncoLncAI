"""Optional, provenance-preserving lncRNA localization tool."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import importlib.metadata
import subprocess
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oncolncai.schemas import Provenance, ResultStatus, ToolError, ToolResult


class LocalizationCompartment(StrEnum):
    NUCLEAR = "nuclear"
    CYTOPLASMIC = "cytoplasmic"
    NUCLEAR_AND_CYTOPLASMIC = "nuclear_and_cytoplasmic"
    OTHER = "other"


class LocalizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    lncrna_id: str = Field(min_length=1, max_length=200)
    sequence: str | None = Field(default=None, min_length=1, max_length=200_000)


class LocalizationProbability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compartment: LocalizationCompartment
    probability: float = Field(ge=0.0, le=1.0)


class LocalizationModelProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    artifact_uri: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LocalizationPrediction(BaseModel):
    """Localization context; it contains no biomarker score or reward field."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    lncrna_id: str = Field(min_length=1)
    predicted_compartment: LocalizationCompartment
    probabilities: tuple[LocalizationProbability, ...] | None = None
    sequence_length: int | None = Field(default=None, gt=0)
    model: LocalizationModelProvenance

    @model_validator(mode="after")
    def validate_probabilities(self) -> LocalizationPrediction:
        if self.probabilities is None:
            return self
        compartments = [item.compartment for item in self.probabilities]
        if len(compartments) != len(set(compartments)):
            raise ValueError("localization probability compartments must be unique")
        total = sum(item.probability for item in self.probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("localization probabilities must sum to one")
        if self.predicted_compartment not in compartments:
            raise ValueError("predicted compartment must have a reported probability")
        return self


class LocalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: LocalizationPrediction
    biological_context_only: bool = True
    interpretation_limits: tuple[str, ...] = (
        "Localization is biological context, not proof of biomarker quality.",
        "No compartment is automatically favorable or unfavorable.",
    )


class LocalizationBackend(Protocol):
    def predict(self, request: LocalizationRequest) -> LocalizationPrediction: ...


class PrecomputedLocalizationBackend:
    """Read validated, versioned predictions without requiring a model runtime."""

    def __init__(self, predictions: Mapping[str, LocalizationPrediction]) -> None:
        self._predictions = dict(predictions)

    @classmethod
    def from_json(cls, path: str | Path) -> PrecomputedLocalizationBackend:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = tuple(LocalizationPrediction.model_validate(item) for item in payload)
        return cls({item.lncrna_id.casefold(): item for item in records})

    def predict(self, request: LocalizationRequest) -> LocalizationPrediction:
        try:
            return self._predictions[request.lncrna_id.casefold()]
        except KeyError as exc:
            raise LookupError(f"no precomputed localization for {request.lncrna_id}") from exc


class CallableLocalizationBackend:
    """Adapt an existing sequence predictor such as ``predict_probabilities``."""

    def __init__(
        self,
        predictor: Callable[[str], Mapping[str, Any]],
        *,
        model: LocalizationModelProvenance,
    ) -> None:
        self._predictor = predictor
        self._model = model

    def predict(self, request: LocalizationRequest) -> LocalizationPrediction:
        if request.sequence is None:
            raise ValueError("the configured localization predictor requires a sequence")
        raw = self._predictor(request.sequence)
        compartment = LocalizationCompartment(str(raw["predicted_localization"]).casefold())
        probabilities = None
        if "nuclear_probability" in raw and "cytoplasmic_probability" in raw:
            probabilities = (
                LocalizationProbability(compartment=LocalizationCompartment.NUCLEAR, probability=raw["nuclear_probability"]),
                LocalizationProbability(compartment=LocalizationCompartment.CYTOPLASMIC, probability=raw["cytoplasmic_probability"]),
            )
        return LocalizationPrediction(
            lncrna_id=request.lncrna_id,
            predicted_compartment=compartment,
            probabilities=probabilities,
            sequence_length=raw.get("sequence_length"),
            model=self._model,
        )


class LocalProjectLocalizationBackend:
    """Lazily wrap the versioned ``Deploy_lncrna`` project in place."""

    def __init__(self, project_dir: str | Path, *, python_executable: str | Path | None = None, timeout: float = 60.0) -> None:
        if timeout <= 0:
            raise ValueError("localization timeout must be positive")
        self.project_dir = Path(project_dir).resolve()
        self.python_executable = Path(python_executable).resolve() if python_executable else None
        self.timeout = timeout
        version_path = self.project_dir / "VERSION"
        metadata_path = self.project_dir / "model" / "model_metadata.json"
        artifact_path = self.project_dir / "model" / "random_forest.joblib"
        if not all(path.is_file() for path in (version_path, metadata_path, artifact_path, self.project_dir / "app.py")):
            raise ValueError("localization project is missing app, VERSION, metadata, or model artifact")
        version = version_path.read_text(encoding="utf-8").strip()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_version") != version:
            raise ValueError("localization VERSION and model metadata disagree")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if metadata.get("model_sha256") != digest:
            raise ValueError("localization model artifact hash does not match metadata")
        self.required_sklearn_version = str(metadata.get("scikit_learn_version", ""))
        self.model = LocalizationModelProvenance(
            model_name="lncRNA Localization Random Forest",
            model_version=version,
            source="local Deploy_lncrna project",
            artifact_uri=str(artifact_path),
            artifact_sha256=digest,
        )
        self._delegate: CallableLocalizationBackend | None = None

    def _predict_in_subprocess(self, request: LocalizationRequest) -> LocalizationPrediction:
        if request.sequence is None:
            raise ValueError("the configured localization predictor requires a sequence")
        if self.python_executable is None or not self.python_executable.is_file():
            raise ValueError("configured localization Python executable does not exist")
        script = (
            "import json,sys; "
            "sys.path.insert(0,sys.argv[1]); "
            "from app import predict_probabilities; "
            "print(json.dumps(predict_probabilities(sys.stdin.read())))"
        )
        completed = subprocess.run(
            [str(self.python_executable), "-c", script, str(self.project_dir)],
            input=request.sequence,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "localization subprocess failed"
            raise RuntimeError(message)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("localization subprocess returned no prediction")
        raw = json.loads(lines[-1])
        return CallableLocalizationBackend(lambda sequence: raw, model=self.model).predict(request)

    def _load_delegate(self) -> CallableLocalizationBackend:
        installed_sklearn = importlib.metadata.version("scikit-learn")
        if installed_sklearn != self.required_sklearn_version:
            raise RuntimeError(
                "localization model requires scikit-learn "
                f"{self.required_sklearn_version}, found {installed_sklearn}; "
                "configure python_executable for its pinned environment"
            )
        app_path = self.project_dir / "app.py"
        module_name = f"oncolncai_localization_app_{hashlib.sha256(str(app_path).encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, app_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load the localization application module")
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            sys.path.insert(0, str(self.project_dir))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        predictor = getattr(module, "predict_probabilities", None)
        if not callable(predictor):
            raise RuntimeError("localization app does not expose predict_probabilities")
        return CallableLocalizationBackend(predictor, model=self.model)

    def predict(self, request: LocalizationRequest) -> LocalizationPrediction:
        if self.python_executable is not None:
            return self._predict_in_subprocess(request)
        if self._delegate is None:
            self._delegate = self._load_delegate()
        return self._delegate.predict(request)


def predict_lncrna_localization(
    request: LocalizationRequest,
    *,
    backend: LocalizationBackend | None,
) -> ToolResult[LocalizationResult]:
    """Run an optional backend and preserve its model/version provenance."""

    if backend is None:
        return ToolResult[LocalizationResult](
            status=ResultStatus.FAILURE,
            error=ToolError(
                code="localization_unavailable",
                message="no localization backend is configured",
                retryable=False,
            ),
        )
    try:
        prediction = backend.predict(request)
        if prediction.lncrna_id.casefold() != request.lncrna_id.casefold():
            raise ValueError("localization result identifier does not match the request")
    except LookupError as exc:
        return ToolResult[LocalizationResult](
            status=ResultStatus.FAILURE,
            error=ToolError(code="localization_not_found", message=str(exc), retryable=False),
        )
    except Exception as exc:
        return ToolResult[LocalizationResult](
            status=ResultStatus.FAILURE,
            error=ToolError(code="localization_prediction_failed", message=str(exc) or type(exc).__name__, retryable=False),
        )
    return ToolResult[LocalizationResult](
        status=ResultStatus.SUCCESS,
        data=LocalizationResult(prediction=prediction),
        provenance=[
            Provenance(
                source=prediction.model.source,
                source_id=prediction.lncrna_id,
                dataset_version=prediction.model.model_version,
            )
        ],
        warnings=[
            "Localization is contextual evidence only and does not change biomarker quality or ranking."
        ],
    )
