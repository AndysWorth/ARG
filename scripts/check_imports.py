#!/usr/bin/env python3
"""Verify that every public module in arg/ imports without error.

Formatters and automated tools can silently strip `from __future__ import`
lines or other imports. This script catches those breakages before the test
suite runs.

Exit code 0 on success, 1 if any import fails.
"""

import importlib
import pkgutil
import sys


def main() -> int:
    failed: list[tuple[str, str]] = []
    for mod_info in pkgutil.walk_packages(["arg"], prefix="arg."):
        try:
            importlib.import_module(mod_info.name)
        except Exception as err:
            failed.append((mod_info.name, str(err)))

    if failed:
        for name, msg in failed:
            print(f"FAIL  {name}: {msg}", file=sys.stderr)
        return 1

    print(
        f"All imports OK ({sum(1 for _ in pkgutil.walk_packages(['arg'], prefix='arg.'))} modules)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
