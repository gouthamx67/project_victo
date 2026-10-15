from __future__ import annotations

import os
from pathlib import Path

from src.env_loader import load_dotenv_file
from src.llm_provider import (
    read_api_key_from_env,
    sanitize_api_key,
)


def test_sanitize_api_key_strips_terminal_artifacts():
    assert (
        sanitize_api_key("gsk_abc123\x1b[C")
        == "gsk_abc123"
    )

    assert (
        sanitize_api_key("gsk_abc123\x1b[K")
        == "gsk_abc123"
    )


def test_sanitize_api_key_strips_whitespace_and_quotes():
    assert sanitize_api_key("  gsk_abc123  ") == "gsk_abc123"
    assert sanitize_api_key('"gsk_abc123"') == "gsk_abc123"
    assert sanitize_api_key("'gsk_abc123'") == "gsk_abc123"


def test_sanitize_api_key_returns_none_for_blank_values():
    assert sanitize_api_key(None) is None
    assert sanitize_api_key("") is None
    assert sanitize_api_key("   \n") is None
    assert sanitize_api_key("\x1b[C") is None


def test_read_api_key_from_env_sanitizes(monkeypatch):
    monkeypatch.setenv(
        "GROQ_API_KEY",
        " gsk_abc123\x1b[C ",
    )

    assert (
        read_api_key_from_env("GROQ_API_KEY")
        == "gsk_abc123"
    )


def test_load_dotenv_file_applies_missing_variables(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("UNIT_TEST_KEY", raising=False)

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "# comment line\n"
        "\n"
        "UNIT_TEST_KEY=value-1\n",
        encoding="utf-8",
    )

    applied = load_dotenv_file(dotenv_path)

    assert applied == {"UNIT_TEST_KEY": "value-1"}
    assert os.environ["UNIT_TEST_KEY"] == "value-1"


def test_load_dotenv_file_keeps_existing_non_empty_values(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("UNIT_TEST_KEY", "from-shell")

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "UNIT_TEST_KEY=from-file\n",
        encoding="utf-8",
    )

    applied = load_dotenv_file(dotenv_path)

    assert applied == {}
    assert os.environ["UNIT_TEST_KEY"] == "from-shell"


def test_load_dotenv_file_fills_empty_placeholder(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("UNIT_TEST_KEY", "")

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "UNIT_TEST_KEY=from-file\n",
        encoding="utf-8",
    )

    applied = load_dotenv_file(dotenv_path)

    assert applied == {"UNIT_TEST_KEY": "from-file"}
    assert os.environ["UNIT_TEST_KEY"] == "from-file"


def test_load_dotenv_file_supports_export_and_quotes(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("UNIT_TEST_QUOTED", raising=False)
    monkeypatch.delenv("UNIT_TEST_SINGLE", raising=False)

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        'export UNIT_TEST_QUOTED="value two"\n'
        "UNIT_TEST_SINGLE='value-three'\n",
        encoding="utf-8",
    )

    applied = load_dotenv_file(dotenv_path)

    assert applied == {
        "UNIT_TEST_QUOTED": "value two",
        "UNIT_TEST_SINGLE": "value-three",
    }


def test_load_dotenv_file_ignores_missing_file(tmp_path: Path):
    assert (
        load_dotenv_file(tmp_path / "missing.env")
        == {}
    )
