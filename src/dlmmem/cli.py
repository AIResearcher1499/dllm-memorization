"""Command line entry point (pattern from the fertility-precision repo)."""

from __future__ import annotations

import sys


def doctor() -> int:
    import importlib

    ok = True
    for mod in ("torch",):
        try:
            m = importlib.import_module(mod)
            print(f"  ok      {mod:<10} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  MISSING {mod:<10} {type(exc).__name__}: {exc}")
    if ok:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  gpu     cuda:{i} {p.name} {p.total_memory // 2**20} MiB")
        elif torch.backends.mps.is_available():
            print("  gpu     mps (dev box)")
        else:
            print("  gpu     NONE (cpu only)")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: dlmmem {doctor|b0|p0} [args...]\n"
              "  doctor  check installs + GPU\n"
              "  b0      Stage B-0 runner (see `dlmmem b0 --help`)\n"
              "  p0      Pilot P0: AR vs MDM instrument probe (docs/pilot-p0.md)")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "doctor":
        return doctor()
    if cmd == "b0":
        from .b0 import main as b0_main

        b0_main(rest)
        return 0
    if cmd == "p0":
        from .p0 import main as p0_main

        p0_main(rest)
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
