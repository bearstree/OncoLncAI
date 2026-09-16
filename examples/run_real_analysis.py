"""Run real genome-wide discovery or single-candidate validation."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from oncolncai.config import load_settings
from oncolncai.discovery_analysis import GenomeWideAnalysisRequest, GenomeWideAnalysisRunner
from oncolncai.gdc_counts import GDCRawCountClient
from oncolncai.pubmed import PubMedClient
from oncolncai.real_analysis import RealAnalysisRequest, RealAnalysisRunner
from oncolncai.tcga_capabilities import GDCCapabilityClient
from oncolncai.tcga_survival import TCGASurvivalDataClient
from oncolncai.tcga_discovery import TCGAGenomeWideClient


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", default="lung adenocarcinoma")
    parser.add_argument("--candidate", default="NEAT1")
    parser.add_argument("--ensembl-id", default="ENSG00000245532")
    parser.add_argument("--question", default="Which lncRNAs are promising biomarkers for lung adenocarcinoma, and are they prognostically relevant?")
    parser.add_argument("--summary", action="store_true", help="Print a compact result instead of full JSON")
    parser.add_argument("--research-mode", choices=("genome-wide", "single-candidate"), default="genome-wide")
    args = parser.parse_args()
    settings = load_settings()
    api_key = settings.ncbi_api_key.get_secret_value() if settings.ncbi_api_key else None
    run_id = datetime.now(timezone.utc).strftime("cli-%Y%m%dT%H%M%SZ")
    if args.research_mode == "genome-wide":
        runner = GenomeWideAnalysisRunner(
            genome_client=TCGAGenomeWideClient(cache_dir=settings.cache_dir / "tcga_discovery"),
            capability_client=GDCCapabilityClient(cache_dir=settings.cache_dir / "gdc"),
            survival_client=TCGASurvivalDataClient(cache_dir=settings.cache_dir / "tcga_survival"),
            pubmed_client=PubMedClient(api_key=api_key, cache_dir=settings.cache_dir / "pubmed"),
            output_directory=settings.cache_dir / "runs",
            raw_count_client=GDCRawCountClient(cache_dir=settings.cache_dir / "gdc_raw_counts"),
        )
        result = runner.run(
            GenomeWideAnalysisRequest(run_id=run_id, question=args.question, cancer_name=args.cancer),
            progress=lambda snapshot: print(f"[{snapshot.fraction:>5.0%}] {snapshot.message}", flush=True),
        )
        if args.summary:
            de_table = next(
                (path for path in result.manifest.output_artifacts if path.endswith("complete_de_table.csv")),
                None,
            )
            print({"status": result.status, "project": result.project_id,
                   "candidates": [item.gene_symbol for item in result.rankings],
                   "de_table": de_table})
        else:
            print(result.model_dump_json(indent=2))
        return

    runner = RealAnalysisRunner(
        pubmed_client=PubMedClient(api_key=api_key, cache_dir=settings.cache_dir / "pubmed"),
        capability_client=GDCCapabilityClient(cache_dir=settings.cache_dir / "gdc"),
        survival_client=TCGASurvivalDataClient(cache_dir=settings.cache_dir / "tcga_survival"),
        manifest_directory=settings.cache_dir / "manifests",
    )
    result = runner.run(
        RealAnalysisRequest(
            run_id=run_id,
            question=args.question,
            cancer_name=args.cancer,
            candidate=args.candidate,
            ensembl_gene_id=args.ensembl_id,
        ),
        progress=lambda fraction, message: print(f"[{fraction:>5.0%}] {message}", flush=True),
    )
    if args.summary:
        survival = result.survival
        print({
            "status": result.status.value,
            "project": result.project_id,
            "candidate": result.candidate,
            "papers": [paper.pmid for paper in result.papers],
            "sample_count": survival.sample_count if survival else None,
            "event_count": survival.event_count if survival else None,
            "hazard_ratio": survival.hazard_ratio if survival else None,
            "cox_p_value": survival.cox_p_value if survival else None,
        })
    else:
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
