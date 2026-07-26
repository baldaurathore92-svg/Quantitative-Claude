"""Command-line entry point.

Modes
-----
``live``
    Angel One SmartAPI V2 SnapQuote feed. Requires credentials and the
    ``smartapi-python`` package.
``replay``
    Replays a recorded JSONL session. Deterministic; needs no credentials.
``synthetic``
    Seeded synthetic book. Exercises the whole pipeline and the console without
    any external dependency. Explicitly not market data.

Examples
--------
::

    python -m snapshot_quant_v4.main --config config.json --mode synthetic
    python -m snapshot_quant_v4.main --config config.json --mode replay \\
        --file recordings/session.jsonl --speed 0
    python -m snapshot_quant_v4.main --config config.json --mode live \\
        --record recordings/session.jsonl

Credentials are read from ``config.json`` or, preferably, from the environment
(``SQ4_API_KEY``, ``SQ4_CLIENT_CODE``, ``SQ4_PIN``, ``SQ4_TOTP_SECRET``). Every
credential is registered with the logging redactor before any other component
starts, so a third-party stack trace cannot leak one.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .adapter.angel_v2 import AdapterUnavailableError, AngelOneAdapter, build_gap_notifier
from .adapter.replay import ReplaySource, SnapshotRecorder
from .adapter.synthetic import SyntheticConfig, SyntheticSource
from .config import AppConfig, ConfigError, RuntimeConfig, load_config
from .engine.quant_engine import EngineRegistry
from .output.console import create_renderer
from .runner import EngineRunner
from .utils.clock import SystemClock
from .utils.logging_utils import configure_logging, get_logger
from .utils.types import MarketDataSource

_LOGGER = get_logger(__name__)

#: Process exit codes, so a supervisor can distinguish failure modes.
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DEPENDENCY = 3
EXIT_RUNTIME = 4
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="snapshot_quant_v4",
        description=(
            "Snapshot Quant Engine V4: a deterministic, snapshot-only quant "
            "engine for the NSE cash market built on SmartAPI SnapQuote (mode 3)."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="path to the JSON configuration file (default: config.json)",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "replay", "synthetic"),
        default="synthetic",
        help=(
            "data source. 'synthetic' is the default because it requires no "
            "credentials and no network"
        ),
    )
    parser.add_argument("--file", help="JSONL recording to replay (--mode replay)")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help=(
            "replay speed: 0 for unpaced, 1.0 to reproduce the original "
            "inter-snapshot intervals (default: 0)"
        ),
    )
    parser.add_argument(
        "--record",
        help="write every received snapshot to this JSONL file for later replay",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "rich", "plain", "none"),
        help="override runtime.renderer from the configuration",
    )
    parser.add_argument(
        "--fps", type=float, help="override runtime.render_fps from the configuration"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        help="stop after this many snapshots (0 means unlimited)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2_000,
        help="number of synthetic snapshots to generate (--mode synthetic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_726,
        help="synthetic generator seed (--mode synthetic)",
    )
    parser.add_argument(
        "--paced-synthetic",
        action="store_true",
        help="pace synthetic snapshots in real time so the console is watchable",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="override runtime.log_level",
    )
    parser.add_argument(
        "--log-console",
        action="store_true",
        help=(
            "also write logs to stderr. Off by default because the renderer owns "
            "the terminal and interleaved log lines corrupt the display"
        ),
    )
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides to the loaded configuration.

    Overrides are applied by constructing a new configuration object rather than
    mutating one, so the immutability guarantee holds for the whole process
    lifetime.
    """
    current = config.runtime
    runtime = RuntimeConfig(
        queue_size=current.queue_size,
        render_fps=current.render_fps if args.fps is None else float(args.fps),
        log_level=current.log_level if args.log_level is None else args.log_level,
        log_file=current.log_file,
        log_console=current.log_console or bool(args.log_console),
        latency_window=current.latency_window,
        stats_interval_ms=current.stats_interval_ms,
        max_snapshots=(
            current.max_snapshots
            if args.max_snapshots is None
            else int(args.max_snapshots)
        ),
        renderer=current.renderer if args.renderer is None else args.renderer,
    )
    if runtime == current:
        return config
    return dataclasses.replace(config, runtime=runtime)


def _build_source(
    config: AppConfig, args: argparse.Namespace, registry: EngineRegistry
) -> MarketDataSource:
    """Construct the data source selected by ``--mode``.

    Raises
    ------
    ConfigError
        For a missing or invalid mode-specific argument.
    """
    if args.mode == "live":
        return AngelOneAdapter(
            config,
            gap_callback=build_gap_notifier(registry.notify_gap),
        )
    if args.mode == "replay":
        if not args.file:
            raise ConfigError("--mode replay requires --file")
        return ReplaySource(
            args.file,
            speed=args.speed,
            max_snapshots=config.runtime.max_snapshots,
        )
    symbol = config.symbols[0]
    return SyntheticSource(
        SyntheticConfig(
            token=symbol.token,
            symbol=symbol.symbol,
            exchange_type=symbol.exchange_type,
            tick_paise=max(1, round(symbol.tick_size * 100)),
            count=args.count,
            seed=args.seed,
        ),
        paced=args.paced_synthetic,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the engine. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        config = _apply_overrides(load_config(args.config), args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    logging_handle = configure_logging(
        level=config.runtime.log_level,
        log_file=config.runtime.log_file,
        console=config.runtime.log_console,
        secrets=config.credentials.secret_values(),
    )

    recorder: SnapshotRecorder | None = None
    try:
        clock = SystemClock()
        registry = EngineRegistry(config, clock=clock)
        source = _build_source(config, args, registry)
        renderer = create_renderer(config.runtime.renderer)

        if args.record:
            recorder = SnapshotRecorder(Path(args.record))
            recorder.open()

        _LOGGER.info(
            "starting in %s mode with %d symbol(s): %s",
            args.mode,
            len(config.symbols),
            ", ".join(f"{s.symbol}({s.token})" for s in config.symbols),
        )
        runner = EngineRunner(
            config,
            source,
            registry,
            renderer,
            clock=clock,
            recorder=recorder,
        )
        runner.run()
        for line in runner.summary_lines():
            print(line)
    except ConfigError as exc:
        _LOGGER.error("configuration error: %s", exc)
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except AdapterUnavailableError as exc:
        _LOGGER.error("%s", exc)
        print(str(exc), file=sys.stderr)
        return EXIT_DEPENDENCY
    except FileNotFoundError as exc:
        _LOGGER.error("%s", exc)
        print(str(exc), file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        _LOGGER.info("interrupted by user")
        return EXIT_INTERRUPTED
    except Exception as exc:
        _LOGGER.exception("fatal error")
        print(f"fatal error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    else:
        return EXIT_OK
    finally:
        logging_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
