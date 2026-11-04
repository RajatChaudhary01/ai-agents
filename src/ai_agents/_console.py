"""Console setup shared by the CLI and the web entry point."""

import sys


def use_utf8_output() -> None:
    """Windows consoles default to a legacy code page; switch stdout to UTF-8 for emoji."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
