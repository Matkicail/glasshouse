"""``glasshouse`` on the command line. One verb so far: ``bench``."""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``glasshouse`` script."""
    parser = argparse.ArgumentParser(
        prog="glasshouse", description="interpretable ML, honestly scored"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="run a named benchmark and write its report")
    bench.add_argument("name", help="benchmark name (see `glasshouse bench --list`)")
    bench.add_argument("--out", default="benchmarks", help="directory for <name>/report.{json,md}")
    sub.add_parser("list", help="list the named benchmarks")
    args = parser.parse_args(argv)

    from glasshouse.benchmarks import BENCHMARKS, run_named  # noqa: PLC0415 — keep --help fast

    if args.command == "list":
        for name, b in BENCHMARKS.items():
            models = ", ".join(m.label for m in b.models)
            print(f"{name:<20} {b.dataset:<16} {b.task.family:<9} models: {models}")
        return 0
    t0 = time.perf_counter()
    result = run_named(args.name)
    out = result.write(f"{args.out}/{args.name}")
    print(result.to_markdown())
    print(
        f"\nwrote {out}/report.json and report.md in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
