"""Latency benchmark for the snapshot decision path.

What is measured
----------------
The wall-clock cost of :meth:`SymbolEngine.process` for one snapshot: validation,
shared statistics, all features, regime, quality, confidence, composite,
threshold and the state machine. That is the path whose latency matters, because
it stands between a snapshot arriving and a decision existing.

What is deliberately *not* measured
-----------------------------------
*   Websocket receive and library-level parsing, which are dominated by network
    and by the third-party client.
*   Terminal rendering and logging, which run on other threads at a fixed frame
    rate and are orders of magnitude more expensive per call. Including them
    would produce a number that describes formatting rather than computation.
*   The construction of the human-readable reason strings, for the same reason;
    they are built after the measured region ends.

Method
------
Snapshots are generated up front from a seeded generator, so generation cost is
excluded and the input is identical between runs. Percentiles are reported rather
than a mean: a mean hides exactly the tail that matters, and the tail here is
dominated by the periodic exact recomputation inside the drift-corrected rolling
estimators, which is an intentional design trade-off and should be visible.

Usage
-----
::

    python -m bench.benchmark
    python -m bench.benchmark --count 50000 --symbols 5 --repeat 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapshot_quant_v4.adapter.synthetic import SyntheticConfig, SyntheticSource
from snapshot_quant_v4.config import config_from_mapping
from snapshot_quant_v4.engine.quant_engine import EngineRegistry
from snapshot_quant_v4.utils.types import RawSnapshot


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing summary for one run."""

    label: str
    samples: int
    mean_us: float
    p50_us: float
    p90_us: float
    p99_us: float
    max_us: float
    throughput_per_second: float

    def format(self) -> str:
        """Return a single aligned report line."""
        return (
            f"{self.label:<22} n={self.samples:<7} "
            f"mean={self.mean_us:7.1f}us  p50={self.p50_us:7.1f}us  "
            f"p90={self.p90_us:7.1f}us  p99={self.p99_us:7.1f}us  "
            f"max={self.max_us:8.1f}us  "
            f"{self.throughput_per_second:9.0f} snapshots/s"
        )


def build_snapshots(count: int, token: str, symbol: str, seed: int) -> list[RawSnapshot]:
    """Generate a deterministic snapshot sequence up front."""
    source = SyntheticSource(
        SyntheticConfig(count=count, token=token, symbol=symbol, seed=seed)
    )
    source.start()
    return list(source.snapshots())


def measure(label: str, registry: EngineRegistry, snapshots: list[RawSnapshot]) -> BenchmarkResult:
    """Process every snapshot, timing each call individually."""
    timings: list[float] = []
    append = timings.append
    process = registry.process
    counter = time.perf_counter_ns

    started = counter()
    for snapshot in snapshots:
        before = counter()
        process(snapshot)
        append((counter() - before) / 1_000.0)
    elapsed_s = (counter() - started) / 1e9

    timings.sort()
    count = len(timings)
    return BenchmarkResult(
        label=label,
        samples=count,
        mean_us=statistics.fmean(timings),
        p50_us=timings[count // 2],
        p90_us=timings[min(count - 1, int(count * 0.90))],
        p99_us=timings[min(count - 1, int(count * 0.99))],
        max_us=timings[-1],
        throughput_per_second=count / elapsed_s if elapsed_s > 0 else 0.0,
    )


def build_registry(symbol_count: int) -> tuple[EngineRegistry, list[tuple[str, str]]]:
    """Build a registry for ``symbol_count`` instruments."""
    instruments = [(f"{3045 + index}", f"SYM{index}") for index in range(symbol_count)]
    config = config_from_mapping(
        {
            "symbols": [
                {"token": token, "symbol": name, "tick_size": 0.05}
                for token, name in instruments
            ],
            "runtime": {"log_file": None, "render_fps": 0.0},
        }
    )
    return EngineRegistry(config), instruments


def run(count: int, symbol_count: int, repeat: int, seed: int) -> list[BenchmarkResult]:
    """Run the benchmark and return one result per repetition."""
    results: list[BenchmarkResult] = []
    for iteration in range(1, repeat + 1):
        registry, instruments = build_registry(symbol_count)
        snapshots: list[RawSnapshot] = []
        per_symbol = max(1, count // symbol_count)
        for index, (token, name) in enumerate(instruments):
            snapshots.extend(build_snapshots(per_symbol, token, name, seed + index))
        # Interleave so multi-symbol runs alternate between engines, which is how
        # a live feed arrives and which exercises cache behaviour realistically.
        if symbol_count > 1:
            snapshots.sort(key=lambda snapshot: snapshot.exchange_timestamp_ms)
        label = f"{symbol_count} symbol(s) run {iteration}"
        results.append(measure(label, registry, snapshots))
    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        prog="bench.benchmark",
        description="Measure per-snapshot decision-path latency.",
    )
    parser.add_argument("--count", type=int, default=20_000, help="snapshots per run")
    parser.add_argument("--symbols", type=int, default=1, help="number of instruments")
    parser.add_argument("--repeat", type=int, default=3, help="repetitions")
    parser.add_argument("--seed", type=int, default=20_260_726, help="generator seed")
    args = parser.parse_args(argv)

    if args.count <= 0 or args.symbols <= 0 or args.repeat <= 0:
        print("count, symbols and repeat must all be positive", file=sys.stderr)
        return 2

    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(
        "measuring SymbolEngine.process: validation, statistics, 10 features, "
        "regime, quality, confidence, composite, threshold, state machine"
    )
    print()
    results = run(args.count, args.symbols, args.repeat, args.seed)
    for result in results:
        print(result.format())
    best = min(results, key=lambda result: result.p50_us)
    print()
    print(
        f"best p50 {best.p50_us:.1f}us, p99 {best.p99_us:.1f}us "
        f"({best.throughput_per_second:.0f} snapshots/s single threaded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
