#!/usr/bin/env python3
"""Evaluate extraction quality from Phase 0 results.

Usage:
    # Schema check only (free, no API key needed):
    python scripts/evaluate_extraction.py results/phase0_extraction.json

    # Full evaluation with LLM-as-judge (uses GPT-4o, costs ~$0.02/paper):
    python scripts/evaluate_extraction.py results/phase0_extraction.json --llm-judge

    # Save detailed report:
    python scripts/evaluate_extraction.py results/phase0_extraction.json --llm-judge --output results/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

from kosa.config import Settings
from kosa.ingestion.evaluate import evaluate_extractions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction quality")
    parser.add_argument("input", help="Path to phase0_extraction.json")
    parser.add_argument("--llm-judge", action="store_true", help="Run GPT-4o LLM-as-judge")
    parser.add_argument("--judge-model", default="gpt-4o", help="Model for LLM judge")
    parser.add_argument("--output", help="Save detailed report JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results)} extraction results")

    client = None
    if args.llm_judge:
        settings = Settings()
        if not settings.openai_api_key:
            print("ERROR: OPENAI_API_KEY needed for --llm-judge", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=settings.openai_api_key)

    report = evaluate_extractions(
        results,
        client=client,
        judge_model=args.judge_model,
        run_llm_judge=args.llm_judge,
    )

    print(report.summary())

    if args.output:
        out = {
            "schema_checks": [
                {
                    "paper_id": c.paper_id,
                    "passed": c.passed,
                    "issues": c.issues,
                }
                for c in report.schema_checks
            ],
        }
        if report.llm_judgments:
            out["llm_judgments"] = [
                {
                    "paper_id": j.paper_id,
                    "entity_accuracy": j.entity_accuracy,
                    "relation_accuracy": j.relation_accuracy,
                    "entity_verdicts": j.entity_verdicts,
                    "relation_verdicts": j.relation_verdicts,
                    "missing_entities": j.missing_entities,
                    "missing_relations": j.missing_relations,
                }
                for j in report.llm_judgments
            ]
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Detailed report saved to {out_path}")


if __name__ == "__main__":
    main()
