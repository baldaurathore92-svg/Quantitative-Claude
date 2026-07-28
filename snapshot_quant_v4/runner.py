"""Process wiring: source, engine, renderer, recorder, shutdown.

Threading model
---------------
::

    feed thread(s)      engine thread (this loop)      render thread
    -------------       -------------------------      -------------
    parse + enqueue --> drain queue                    every 1/fps:
                        engine.process(snapshot)         copy latest outputs
                        store latest output              draw frame
                        record (optional)

Three properties this buys, all of them requirements rather than preferences:

*   **The feed is never blocked by compute.** The websocket callback only parses
    and enqueues.
*   **Compute is never blocked by the terminal.** Rendering happens on its own
    thread at a fixed frame rate. A terminal write costs milliseconds; a snapshot
    costs hundreds of microseconds.
*   **A quiet feed still refreshes the display.** A render loop driven by
    incoming snapshots would freeze the screen exactly when the operator most
    wants to see that nothing is arriving.

Shared state between the engine and render threads is a single dictionary of the
latest output per symbol, guarded by a lock that is held only for a shallow copy.

Shutdown is cooperative: ``SIGINT`` and ``SIGTERM`` set an event, the engine loop
finishes its current snapshot, the source is stopped, the render thread joins, and
the recorder and logging handles are closed in a ``finally`` block.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import FrameType
from typing import Any

from .adapter.replay import SnapshotRecorder
from .buffers.ring_buffer import RingBuffer
from .config import AppConfig
from .engine.quant_engine import EngineRegistry
from .utils.clock import Clock, SystemClock
from .utils.logging_utils import get_logger
from .utils.types import (
    Direction,
    EngineOutput,
    MarketDataSource,
    RawSnapshot,
    Renderer,
)

_LOGGER = get_logger(__name__)

_MIN_LIVE_STATUS_INTERVAL_MS = 5_000.0


class LatencyTracker:
    """Sliding-window latency percentiles for monitoring.

    Percentiles are computed on demand — at the render frame rate, never per
    snapshot — so the sort cost stays off the decision path. Recording a sample is
    a single ring-buffer push.
    """

    __slots__ = ("_buffer",)

    def __init__(self, window: int) -> None:
        self._buffer = RingBuffer(window)

    def record(self, microseconds: float) -> None:
        """Add one sample."""
        self._buffer.push(microseconds)

    def __len__(self) -> int:
        return len(self._buffer)

    def percentiles(self) -> tuple[float, float, float]:
        """Return ``(p50, p99, max)`` in microseconds, or zeros when empty."""
        count = len(self._buffer)
        if count == 0:
            return 0.0, 0.0, 0.0
        samples = sorted(self._buffer.iter_values())
        p50 = samples[count // 2]
        p99 = samples[min(count - 1, int(count * 0.99))]
        return p50, p99, samples[-1]


@dataclass(slots=True)
class RunnerStats:
    """Counters for the footer line and the shutdown summary."""

    processed: int = 0
    outputs: int = 0
    started_monotonic_ms: float = 0.0

    def rate_per_second(self, now_ms: float) -> float:
        """Average processed snapshots per second since start."""
        elapsed_ms = now_ms - self.started_monotonic_ms
        if elapsed_ms <= 0.0:
            return 0.0
        return self.processed * 1000.0 / elapsed_ms


class EngineRunner:
    """Drives a :class:`~engine.quant_engine.EngineRegistry` from a data source.

    Parameters
    ----------
    config:
        Application configuration.
    source:
        Any :class:`~utils.types.MarketDataSource`.
    registry:
        Engine registry. Injected so that a caller can pre-configure engines.
    renderer:
        Any :class:`~utils.types.Renderer`.
    clock:
        Injected clock.
    recorder:
        Optional recorder; when present every accepted raw snapshot is written for
        later replay.
    install_signal_handlers:
        Whether to install ``SIGINT``/``SIGTERM`` handlers. Disabled by tests and
        by embedded use, since handlers are process-global state and can only be
        installed from the main thread.
    live_status:
        Emit rate-limited current-price and model-signal records. Intended for a
        live headless process whose renderer cannot expose in-memory state.
    """

    __slots__ = (
        "_clock",
        "_config",
        "_install_signals",
        "_latency",
        "_latest",
        "_latest_lock",
        "_live_status",
        "_recorder",
        "_registry",
        "_render_thread",
        "_renderer",
        "_source",
        "_stats",
        "_status_thread",
        "_stop",
    )

    def __init__(
        self,
        config: AppConfig,
        source: MarketDataSource,
        registry: EngineRegistry,
        renderer: Renderer,
        *,
        clock: Clock | None = None,
        recorder: SnapshotRecorder | None = None,
        install_signal_handlers: bool = True,
        live_status: bool = False,
    ) -> None:
        self._config = config
        self._source = source
        self._registry = registry
        self._renderer = renderer
        self._clock = clock if clock is not None else SystemClock()
        self._recorder = recorder
        self._install_signals = install_signal_handlers
        self._live_status = live_status
        self._stop = threading.Event()
        self._latest: dict[str, EngineOutput] = {}
        self._latest_lock = threading.Lock()
        self._latency = LatencyTracker(config.runtime.latency_window)
        self._stats = RunnerStats()
        self._render_thread: threading.Thread | None = None
        self._status_thread: threading.Thread | None = None

    # -- introspection ----------------------------------------------------- #

    @property
    def stats(self) -> RunnerStats:
        """Runner counters."""
        return self._stats

    @property
    def latency(self) -> LatencyTracker:
        """Latency tracker."""
        return self._latency

    def request_stop(self) -> None:
        """Ask the runner to finish. Safe to call from any thread."""
        self._stop.set()

    # -- lifecycle --------------------------------------------------------- #

    def run(self) -> RunnerStats:
        """Run until the source is exhausted or a stop is requested."""
        self._stats.started_monotonic_ms = self._clock.monotonic_ms()
        restore = self._install_handlers()
        try:
            self._source.start()
            self._renderer.start()
            self._start_render_thread()
            self._start_status_thread()
            self._consume()
        finally:
            self._stop.set()
            self._shutdown(restore)
        return self._stats

    def _consume(self) -> None:
        """The engine thread: drain the source and process every snapshot."""
        limit = self._config.runtime.max_snapshots
        for raw in self._source.snapshots():
            if self._stop.is_set():
                break
            self._handle(raw)
            if limit and self._stats.processed >= limit:
                _LOGGER.info("reached max_snapshots=%d; stopping", limit)
                break

    def _handle(self, raw: RawSnapshot) -> None:
        """Process one snapshot and publish the result."""
        self._stats.processed += 1
        if self._recorder is not None:
            try:
                self._recorder.write(raw)
            except OSError as exc:
                # A recording failure must not stop trading; it is reported and
                # recording is abandoned for the rest of the session.
                _LOGGER.error("recording failed, disabling recorder: %s", exc)
                self._recorder = None

        output = self._registry.process(raw)
        if output is None:
            return
        self._stats.outputs += 1
        self._latency.record(output.compute_us)
        with self._latest_lock:
            self._latest[output.token] = output

    # -- rendering --------------------------------------------------------- #

    def _start_render_thread(self) -> None:
        """Start the fixed-rate render loop."""
        if self._config.runtime.render_fps <= 0.0:
            return
        thread = threading.Thread(target=self._render_loop, name="renderer", daemon=True)
        self._render_thread = thread
        thread.start()

    def _render_loop(self) -> None:
        """Draw a frame at the configured rate until stopped."""
        interval = 1.0 / self._config.runtime.render_fps
        while not self._stop.wait(interval):
            try:
                self._draw()
            except Exception:
                _LOGGER.exception("render failed")
                return
        # One final frame so the last state is visible after shutdown.
        try:
            self._draw()
        except Exception:
            _LOGGER.exception("final render failed")

    def _draw(self) -> None:
        """Render the current snapshot of outputs."""
        with self._latest_lock:
            outputs = list(self._latest.values())
        if not outputs:
            return
        outputs.sort(key=lambda output: output.symbol)
        self._renderer.render(outputs, self.footer())

    def footer(self) -> str:
        """Build the monitoring footer line."""
        p50, p99, worst = self._latency.percentiles()
        now_ms = self._clock.monotonic_ms()
        parts = [
            f"processed {self._stats.processed}",
            f"outputs {self._stats.outputs}",
            f"{self._stats.rate_per_second(now_ms):.1f}/s",
            f"compute p50 {p50:.0f}us p99 {p99:.0f}us max {worst:.0f}us",
        ]
        parts.extend(self._source_metrics())
        for token, stats in self._registry.statistics().items():
            parts.append(
                f"{token}: acc {stats.accepted} rej {stats.rejected} "
                f"blk {stats.blocked} gap {stats.gaps}"
            )
        return " | ".join(parts)

    def _source_metrics(self) -> list[str]:
        """Collect optional metrics that a source may expose.

        Sources are only required to implement the three protocol methods, so the
        extra counters are read defensively rather than being made mandatory.
        """
        metrics: list[str] = []
        source: Any = self._source
        dropped = getattr(source, "dropped", None)
        if isinstance(dropped, int):
            metrics.append(f"dropped {dropped}")
        depth = getattr(source, "queue_depth", None)
        if isinstance(depth, int):
            metrics.append(f"queue {depth}")
        reconnects = getattr(source, "reconnects", None)
        if isinstance(reconnects, int):
            metrics.append(f"reconnects {reconnects}")
        errors = getattr(source, "parse_errors", None)
        if isinstance(errors, int) and errors:
            metrics.append(f"parse-errors {errors}")
        return metrics

    # -- live headless status --------------------------------------------- #

    def _status_interval_s(self) -> float:
        """Return the configured interval with a log-safety lower bound."""
        effective_ms = max(
            self._config.runtime.stats_interval_ms,
            _MIN_LIVE_STATUS_INTERVAL_MS,
        )
        return effective_ms / 1000.0

    def _start_status_thread(self) -> None:
        """Start rate-limited journal status output for a live headless run."""
        if not self._live_status:
            return
        interval_s = self._status_interval_s()
        _LOGGER.info(
            "live model status enabled every %.1fs; BUY/SELL means the current "
            "model position bias, WAIT means the model is flat, and no broker "
            "orders are placed",
            interval_s,
        )
        thread = threading.Thread(
            target=self._status_loop,
            name="live-status",
            daemon=True,
        )
        self._status_thread = thread
        thread.start()

    def _status_loop(self) -> None:
        """Publish one current-state record per configured symbol periodically."""
        interval_s = self._status_interval_s()
        while not self._stop.wait(interval_s):
            self._log_current_status(event="periodic")

    def _log_current_status(self, *, event: str) -> None:
        """Take one coherent latest-output snapshot and publish every symbol."""
        with self._latest_lock:
            latest = dict(self._latest)
        for symbol in self._config.symbols:
            current = latest.get(symbol.token)
            if current is None:
                self._log_waiting_symbol(symbol.symbol, symbol.token, event=event)
            else:
                self._log_status_output(current, event=event)

    @staticmethod
    def _model_signal(output: EngineOutput) -> tuple[str, str]:
        """Return the current model-position bias, never an order instruction."""
        position = output.position
        if position is not None and position.is_open:
            if position.direction is Direction.LONG:
                return "BUY", "POSITION"
            if position.direction is Direction.SHORT:
                return "SELL", "POSITION"
        return "WAIT", "FLAT"

    def _log_waiting_symbol(self, symbol: str, token: str, *, event: str) -> None:
        """Keep configured symbols visible even before their first good tick."""
        _LOGGER.warning(
            "LIVE STATUS event=%s symbol=%s token=%s ltp_rupees=n/a "
            "bid_rupees=n/a bid_qty=n/a ask_rupees=n/a ask_qty=n/a "
            "spread_rupees=n/a spread_ticks=n/a spread_bps=n/a "
            "spread_limit_ticks=n/a tick_size_rupees=n/a "
            "model_signal=WAIT signal_type=NO_DATA state=NO_DATA position=FLAT "
            "score=n/a confidence=0%% quality=NO_ACCEPTED_SNAPSHOT "
            "snapshot_index=0 exchange_ms=0 data_age=n/a processed=%d",
            event,
            symbol,
            token,
            self._stats.processed,
        )

    def _log_status_output(self, output: EngineOutput, *, event: str) -> None:
        """Log one immutable engine output without changing decision state."""
        position = output.position
        if position is not None and position.is_open:
            position_text = (
                f"{position.direction.name}:{position.quantity}@{position.entry_price:.2f}"
            )
        else:
            position_text = "FLAT"
        quality = (
            "OK"
            if output.quality.tradable
            else "+".join(reason.value for reason in output.quality.reasons)
        )
        score = (
            f"{output.composite.smoothed:+.3f}"
            if output.composite.valid
            else "n/a"
        )
        model_signal, signal_type = self._model_signal(output)
        spread_bps = output.quality.spread_bps
        if spread_bps is None:
            spread_bps = output.snapshot.spread * 10_000.0 / output.snapshot.mid
        spread_limit_ticks = output.quality.spread_limit_ticks
        if spread_limit_ticks is None:
            spread_limit_ticks = self._config.quality.max_signal_spread_ticks
        data_age_s = max(
            0.0,
            (
                self._clock.monotonic_ms()
                - output.snapshot.received_monotonic_ms
            )
            / 1000.0,
        )
        _LOGGER.info(
            "LIVE STATUS event=%s symbol=%s token=%s ltp_rupees=%.6f "
            "bid_rupees=%.6f bid_qty=%d ask_rupees=%.6f ask_qty=%d "
            "spread_rupees=%.6f spread_ticks=%.6f spread_bps=%.6f "
            "spread_limit_ticks=%.6f tick_size_rupees=%.6f "
            "model_signal=%s signal_type=%s state=%s position=%s score=%s "
            "confidence=%.0f%% quality=%s snapshot_index=%d exchange_ms=%d "
            "data_age=%.1fs",
            event,
            output.symbol,
            output.token,
            output.snapshot.last_traded_price,
            output.snapshot.best_bid.price,
            output.snapshot.best_bid.quantity,
            output.snapshot.best_ask.price,
            output.snapshot.best_ask.quantity,
            output.snapshot.spread,
            output.snapshot.spread_ticks,
            spread_bps,
            spread_limit_ticks,
            output.snapshot.tick_size,
            model_signal,
            signal_type,
            output.state.value,
            position_text,
            score,
            output.composite.confidence * 100.0,
            quality,
            output.snapshot_index,
            output.exchange_timestamp_ms,
            data_age_s,
        )

    # -- shutdown ---------------------------------------------------------- #

    def _install_handlers(self) -> Mapping[int, Any] | None:
        """Install termination handlers, returning the previous ones."""
        if not self._install_signals:
            return None
        previous: dict[int, Any] = {}

        def _handler(signum: int, _frame: FrameType | None) -> None:
            _LOGGER.info("received signal %d; shutting down", signum)
            self.request_stop()
            self._source.stop()

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[signum] = signal.signal(signum, _handler)
            except (ValueError, OSError) as exc:
                # Only the main thread may install handlers; embedded use is fine
                # without them.
                _LOGGER.debug("could not install handler for signal %d: %s", signum, exc)
        return previous

    def _shutdown(self, restore: Mapping[int, Any] | None) -> None:
        """Stop everything in a deterministic order."""
        try:
            self._source.stop()
        except Exception:
            _LOGGER.exception("error stopping source")

        status_thread = self._status_thread
        if status_thread is not None and status_thread.is_alive():
            # The monitor only performs bounded QueueHandler writes, so waiting
            # for it guarantees no periodic record can follow the final snapshot.
            status_thread.join()
        self._status_thread = None
        if self._live_status:
            self._log_current_status(event="shutdown")

        thread = self._render_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._render_thread = None

        try:
            self._renderer.stop()
        except Exception:
            _LOGGER.exception("error stopping renderer")

        if self._recorder is not None:
            self._recorder.close()

        if restore:
            for signum, handler in restore.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):  # pragma: no cover - best effort
                    _LOGGER.debug("could not restore handler for signal %d", signum)

        self._log_summary()

    def _log_summary(self) -> None:
        """Log the end-of-run summary."""
        p50, p99, worst = self._latency.percentiles()
        _LOGGER.info(
            "run complete: processed=%d outputs=%d compute p50=%.0fus p99=%.0fus "
            "max=%.0fus",
            self._stats.processed,
            self._stats.outputs,
            p50,
            p99,
            worst,
        )
        for token, stats in self._registry.statistics().items():
            _LOGGER.info(
                "%s: accepted=%d rejected=%d blocked=%d gaps=%d resets=%d rejects=%s",
                token,
                stats.accepted,
                stats.rejected,
                stats.blocked,
                stats.gaps,
                stats.resets,
                dict(stats.rejects_by_reason) or "{}",
            )

    def summary_lines(self) -> tuple[str, ...]:
        """Return the summary as text, for a caller that wants to print it."""
        p50, p99, worst = self._latency.percentiles()
        lines = [
            f"processed {self._stats.processed} snapshot(s), "
            f"{self._stats.outputs} engine output(s)",
            f"compute latency p50 {p50:.1f}us p99 {p99:.1f}us max {worst:.1f}us",
        ]
        for token, stats in self._registry.statistics().items():
            lines.append(
                f"{token}: accepted {stats.accepted}, rejected {stats.rejected}, "
                f"blocked {stats.blocked}, gaps {stats.gaps}, resets {stats.resets}"
            )
            if stats.rejects_by_reason:
                lines.append(f"    rejects: {dict(stats.rejects_by_reason)}")
        return tuple(lines)


def drain_outputs(
    registry: EngineRegistry, snapshots: Iterable[RawSnapshot]
) -> list[EngineOutput]:
    """Process an iterable of snapshots and collect every output.

    A convenience for tests, benchmarks and research scripts that want the whole
    pipeline without the threading, the renderer or the signal handling.
    """
    outputs: list[EngineOutput] = []
    for raw in snapshots:
        output = registry.process(raw)
        if output is not None:
            outputs.append(output)
    return outputs


def wall_clock_timestamp() -> str:
    """Return a filesystem-safe local timestamp, used for default file names."""
    return time.strftime("%Y%m%d-%H%M%S")


__all__ = [
    "EngineRunner",
    "LatencyTracker",
    "RunnerStats",
    "drain_outputs",
    "wall_clock_timestamp",
]
