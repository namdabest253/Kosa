"""Smoke tests to verify package structure and imports."""

import importlib


def test_import_kosa():
    import kosa

    assert kosa.__version__ == "0.1.0"


def test_import_submodules():
    for module in [
        "kosa.ingestion",
        "kosa.graph",
        "kosa.activation",
        "kosa.agents",
        "kosa.ranking",
        "kosa.entity_resolution",
    ]:
        importlib.import_module(module)


def test_settings_defaults():
    from kosa.config import Settings

    s = Settings(openai_api_key="test", neo4j_password="test")
    assert s.extraction_model == "gpt-4o-mini"
    assert s.hypothesis_model == "gpt-4o"
    assert s.neo4j_uri == "bolt://localhost:7687"
