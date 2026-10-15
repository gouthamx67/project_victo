from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(
    path: str | Path,
) -> dict[str, str]:
    """
    Load ``KEY=VALUE`` pairs from a ``.env``-style file into the
    process environment.

    Behavior:

    - Blank lines and lines starting with ``#`` are ignored.
    - A leading ``export`` keyword is accepted.
    - Values may be wrapped in single or double quotes.
    - Real environment variables take precedence, except that a
      variable that is present but empty is treated as unset so an
      empty placeholder cannot mask a configured value.

    Returns the mapping of variables that were actually applied.
    """

    dotenv_path = Path(path)

    if not dotenv_path.is_file():
        return {}

    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    applied: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()

        if not key:
            continue

        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value

    return applied
