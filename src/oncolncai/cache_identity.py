"""Compatibility identity for reusable validated REAL-data artifacts."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json


class CacheCompatibilityError(ValueError):
    pass


class RealCacheIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str
    source_dataset_id: str
    source_dataset_checksum_or_version: str
    expression_representation: str
    analysis_configuration: dict[str, object]
    de_method: str
    statistical_thresholds: dict[str, float]
    annotation_version: str
    schema_version: str = "1"
    project_version: str
    software_versions: dict[str, str]
    source_git_sha: str


def cache_identity_hash(identity: RealCacheIdentity) -> str:
    canonical = to_json(identity.model_dump(mode="json"), by_alias=True, exclude_none=False,
                        indent=None).decode("utf-8")
    # pydantic-core preserves model field order; sort recursively for a stable public contract.
    import json
    canonical = json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_cached_real_artifact(requested: RealCacheIdentity,
                                  cached: RealCacheIdentity) -> str:
    differences = [name for name in type(requested).model_fields
                   if getattr(requested, name) != getattr(cached, name)]
    if differences:
        raise CacheCompatibilityError(
            "incompatible cached REAL artifact fields: " + ", ".join(differences)
        )
    return "REUSED_VALIDATED_REAL_ARTIFACT"
