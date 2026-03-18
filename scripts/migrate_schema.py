#!/usr/bin/env python3
"""Apply the Kosa Neo4j schema: constraints, indexes, and vector indexes.

Usage:
    python scripts/migrate_schema.py [--vector-dims 1536]

Requires NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars (or .env file).
"""

from __future__ import annotations

import argparse
import sys

from neo4j import GraphDatabase

from kosa.config import Settings
from kosa.graph.schema import (
    NodeLabel,
    SchemaReport,
    get_migration_statements,
    get_vector_index_statement,
)


def run_migration(uri: str, user: str, password: str, vector_dims: int = 1536) -> SchemaReport:
    """Connect to Neo4j and apply all schema statements."""
    report = SchemaReport()

    statements = get_migration_statements()
    # Add vector indexes for each node label
    for label in NodeLabel:
        statements.append(get_vector_index_statement(label.value, dimensions=vector_dims))

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                    if "CONSTRAINT" in stmt:
                        report.constraints_created += 1
                    else:
                        report.indexes_created += 1
                except Exception as e:
                    report.errors.append(f"{stmt[:60]}... → {e}")
    finally:
        driver.close()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Kosa Neo4j schema")
    parser.add_argument("--vector-dims", type=int, default=1536, help="Embedding dimensions")
    args = parser.parse_args()

    settings = Settings()
    if not settings.neo4j_password:
        print("ERROR: NEO4J_PASSWORD not set. Configure .env or environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {settings.neo4j_uri} ...")
    report = run_migration(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, args.vector_dims
    )

    print(f"Constraints created: {report.constraints_created}")
    print(f"Indexes created: {report.indexes_created}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for err in report.errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Schema migration complete.")


if __name__ == "__main__":
    main()
