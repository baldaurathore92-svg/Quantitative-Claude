"""Angel One SmartAPI V2 SnapQuote market-data adapter.

Responsibilities
----------------
Authenticate, subscribe in SnapQuote (mode 3), translate every callback payload
into a :class:`~snapshot_quant_v4.utils.types.RawSnapshot`, and keep the
connection alive. Nothing else: parsing lives in :mod:`adapter.parsing` and the
queue in :mod:`adapter.queueing`, so this module contains only connection
management.

Reconnection
------------
The supervisor is a **loop**, never recursion. A recursive reconnect grows the
stack for the lifetime of the process and fails precisely during the long,
flapping outage where reliability matters most. Backoff is exponential with
jitter; the jitter matters because every client of a broker's feed reconnects at
the same instant after a server-side restart, and a synchronised retry storm is
how a recovering endpoint is knocked over again.

On every successful reconnect the adapter re-subscribes and notifies its gap
callback. That notification is not cosmetic: the engine's rolling estimators span
the outage, and statistics computed across a discontinuity are invalid. The
engine responds by closing any open position and restarting its warmup.

Optional dependency
-------------------
``smartapi-python`` is imported lazily. The engine core, the replay source, the
tests and the benchmark all run without it; only :meth:`AngelOneAdapter.start`
requires it, and it raises :class:`AdapterUnavailableError` with an actionable
message when it is missing. That keeps the production dependency surface off the
research path.

Threading model
---------------
``start`` spawns one supervisor thread which owns the websocket. Callbacks run on
the library's thread and only touch the queue and a few counters. ``snapshots``
is consumed by the engine thread. No lock is held while user code runs.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Final

from ..config import AdapterConfig, AppConfig, CredentialsConfig, SymbolConfig
from ..utils.logging_utils import get_logger
from ..utils.totp import generate_totp, seconds_until_next_step
from ..utils.types import RawSnapshot
from .parsing import PayloadError, coerce_payloads, is_heartbeat, parse_snapshot
from .queueing import SnapshotQueue

_LOGGER = get_logger(__name__)

#: Seconds the supervisor waits between liveness checks while connected.
_SUPERVISOR_POLL_S: Final[float] = 0.5


class AdapterUnavailableError(RuntimeError):
    """Raised when the SmartAPI client library is not installed."""


class LoginError(RuntimeError):
    """Raised when authentication fails after all configured attempts."""


def _import_smartapi() -> tuple[Any, Any]:
    """Import the SmartAPI client lazily.

    Several distribution versions expose different module paths, so each known
    location is tried before giving up.

    Raises
    ------
    AdapterUnavailableError
        If no known import path resolves.
    """
    errors: list[str] = []
    for connect_path, socket_path in (
        ("SmartApi.smartConnect", "SmartApi.smartWebSocketV2"),
        ("smartapi.smartConnect", "smartapi.smartWebSocketV2"),
    ):
        try:
            connect_module = __import__(connect_path, fromlist=["SmartConnect"])
            socket_module = __import__(socket_path, fromlist=["SmartWebSocketV2"])
        except (ImportError, AttributeError) as exc:
            errors.append(f"{connect_path}: {exc}")
        else:
            return connect_module.SmartConnect, socket_module.SmartWebSocketV2
    raise AdapterUnavailableError(
        "the Angel One SmartAPI client is not installed. Install it with "
        "'pip install smartapi-python websocket-client' to run in live mode, or "
        "use replay mode, which needs no third-party packages. Tried: "
        + "; ".join(errors)
    )


class AngelOneAdapter:
    """Live SnapQuote source.

    Implements :class:`~snapshot_quant_v4.utils.types.MarketDataSource`.

    Parameters
    ----------
    config:
        Full application configuration; the adapter reads credentials, symbols
        and its own section.
    gap_callback:
        Invoked with a human-readable reason after every reconnect, so the engine
        can invalidate statistics that span the outage.
    queue_size:
        Override for the hand-off queue capacity. Defaults to
        ``config.runtime.queue_size``.
    """

    __slots__ = (
        "_adapter_config",
        "_connected_event",
        "_credentials",
        "_gap_callback",
        "_heartbeats",
        "_last_message_monotonic_ms",
        "_lock",
        "_messages",
        "_parse_errors",
        "_queue",
        "_sessions",
        "_smart_connect_factory",
        "_socket",
        "_socket_factory",
        "_stop_event",
        "_symbols",
        "_thread",
    )

    def __init__(
        self,
        config: AppConfig,
        *,
        gap_callback: Callable[[str], None] | None = None,
        queue_size: int | None = None,
        smart_connect_factory: Callable[..., Any] | None = None,
        socket_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not config.symbols:
            raise ValueError("adapter requires at least one configured symbol")
        self._credentials: CredentialsConfig = config.credentials
        self._adapter_config: AdapterConfig = config.adapter
        self._symbols: tuple[SymbolConfig, ...] = config.symbols
        self._queue = SnapshotQueue(
            queue_size if queue_size is not None else config.runtime.queue_size
        )
        self._gap_callback = gap_callback
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        # Counting *sessions* rather than reconnects directly avoids the subtle
        # bug of trying to infer "is this the first connection?" from the
        # connected flag, which is cleared on every disconnect.
        self._sessions = 0
        self._parse_errors = 0
        self._heartbeats = 0
        self._messages = 0
        self._last_message_monotonic_ms = 0.0
        self._lock = threading.Lock()
        # Injectable factories keep the connection logic testable without a
        # network: a test supplies fakes and drives the whole lifecycle.
        self._smart_connect_factory = smart_connect_factory
        self._socket_factory = socket_factory

    # -- introspection ----------------------------------------------------- #

    @property
    def sessions(self) -> int:
        """Number of websocket sessions established, including the first."""
        return self._sessions

    @property
    def reconnects(self) -> int:
        """Number of sessions after the first one."""
        return max(0, self._sessions - 1)

    @property
    def parse_errors(self) -> int:
        """Number of payloads that could not be parsed."""
        return self._parse_errors

    @property
    def heartbeats(self) -> int:
        """Number of keep-alive frames received."""
        return self._heartbeats

    @property
    def messages(self) -> int:
        """Number of market-data payloads accepted."""
        return self._messages

    @property
    def dropped(self) -> int:
        """Number of snapshots dropped by the hand-off queue."""
        return self._queue.dropped

    @property
    def queue_depth(self) -> int:
        """Approximate hand-off queue depth."""
        return self._queue.depth

    @property
    def connected(self) -> bool:
        """Whether a websocket session is currently established."""
        return self._connected_event.is_set()

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Authenticate and start the supervisor thread.

        Raises
        ------
        AdapterUnavailableError
            If the SmartAPI client is not installed.
        ConfigError
            If credentials are incomplete.
        """
        if self._thread is not None:
            raise RuntimeError("adapter already started")
        self._credentials.validate_for_live()
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._supervise, name="angel-v2-feed", daemon=True
        )
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the supervisor, close the socket and release the consumer."""
        self._stop_event.set()
        socket = self._socket
        if socket is not None:
            try:
                socket.close_connection()
            except Exception as exc:  # noqa: BLE001 - shutdown must never fail
                # A failure while closing must never mask the shutdown path.
                _LOGGER.warning("error closing websocket: %s", exc)
        self._queue.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                _LOGGER.warning("feed thread did not exit within 5s")
        self._thread = None

    def snapshots(self) -> Iterator[RawSnapshot]:
        """Yield snapshots until :meth:`stop` is called."""
        return self._queue.drain()

    # -- supervisor -------------------------------------------------------- #

    def _supervise(self) -> None:
        """Connect, and reconnect with bounded exponential backoff and jitter."""
        config = self._adapter_config
        delay_ms = config.reconnect_initial_ms
        attempt = 0

        while not self._stop_event.is_set():
            attempt += 1
            try:
                session = self._login()
                self._run_socket(session)
                # A clean return means the socket closed. Reset the backoff so a
                # single transient close does not inherit a long delay from an
                # earlier outage.
                delay_ms = config.reconnect_initial_ms
            except AdapterUnavailableError:
                raise
            except LoginError as exc:
                _LOGGER.error("login failed: %s", exc)
            except Exception:
                _LOGGER.exception("feed session ended with an error")
            finally:
                self._connected_event.clear()

            if self._stop_event.is_set():
                break
            if config.max_reconnect_attempts and attempt >= config.max_reconnect_attempts:
                _LOGGER.error(
                    "giving up after %d reconnect attempts", config.max_reconnect_attempts
                )
                break

            wait_s = self._backoff_seconds(delay_ms)
            _LOGGER.warning("reconnecting in %.2fs (attempt %d)", wait_s, attempt)
            if self._stop_event.wait(wait_s):
                break
            delay_ms = min(delay_ms * config.reconnect_multiplier, config.reconnect_max_ms)

        self._queue.close()
        _LOGGER.info("feed supervisor exited")

    def _backoff_seconds(self, delay_ms: float) -> float:
        """Return the next backoff delay with symmetric jitter applied."""
        jitter = self._adapter_config.reconnect_jitter_ratio
        if jitter <= 0.0:
            return delay_ms / 1000.0
        low = delay_ms * (1.0 - jitter)
        high = delay_ms * (1.0 + jitter)
        return random.uniform(low, high) / 1000.0

    # -- authentication ---------------------------------------------------- #

    def _login(self) -> Mapping[str, str]:
        """Authenticate and return the tokens required by the websocket.

        Retries on transient failures. The TOTP is regenerated on each attempt,
        and generation waits out a code that is about to expire: submitting a code
        in its final second is a common cause of apparently random login
        failures.
        """
        config = self._adapter_config
        credentials = self._credentials
        connect_factory = self._smart_connect_factory
        if connect_factory is None:
            connect_factory, _ = _import_smartapi()

        last_error: Exception | None = None
        for attempt in range(1, config.login_retry_attempts + 1):
            try:
                remaining = seconds_until_next_step()
                if remaining < config.totp_min_remaining_s:
                    time.sleep(remaining + 0.1)
                totp = generate_totp(credentials.totp_secret)
                client = connect_factory(api_key=credentials.api_key)
                response = client.generateSession(
                    credentials.client_code, credentials.pin, totp
                )
                tokens = self._extract_tokens(client, response)
            except Exception as exc:  # noqa: BLE001 - any client failure is retried
                last_error = exc
                _LOGGER.warning(
                    "login attempt %d/%d failed: %s",
                    attempt,
                    config.login_retry_attempts,
                    exc,
                )
                if attempt < config.login_retry_attempts and not self._stop_event.wait(
                    1.0
                ):
                    continue
                break
            else:
                _LOGGER.info(
                    "authenticated as %s (attempt %d)", credentials.client_code, attempt
                )
                return tokens
        raise LoginError(str(last_error) if last_error else "unknown login failure")

    @staticmethod
    def _extract_tokens(client: Any, response: Any) -> Mapping[str, str]:
        """Pull the auth and feed tokens out of a session response.

        The response shape has varied between client versions, so the extraction
        is defensive and reports precisely what was missing.
        """
        if not isinstance(response, Mapping):
            raise LoginError(f"unexpected session response type {type(response).__name__}")
        if response.get("status") is False:
            raise LoginError(str(response.get("message") or "session rejected"))
        data = response.get("data")
        if not isinstance(data, Mapping):
            raise LoginError("session response carried no data block")
        auth_token = data.get("jwtToken") or data.get("access_token")
        refresh_token = data.get("refreshToken") or data.get("refresh_token")
        if not auth_token:
            raise LoginError("session response carried no jwtToken")
        feed_token = data.get("feedToken")
        if not feed_token:
            getter = getattr(client, "getfeedToken", None)
            feed_token = getter() if callable(getter) else None
        if not feed_token:
            raise LoginError("could not obtain a feed token")
        return {
            "auth_token": str(auth_token),
            "refresh_token": str(refresh_token or ""),
            "feed_token": str(feed_token),
        }

    # -- websocket --------------------------------------------------------- #

    def _run_socket(self, session: Mapping[str, str]) -> None:
        """Create the websocket, wire the callbacks and block until it closes."""
        socket_factory = self._socket_factory
        if socket_factory is None:
            _, socket_factory = _import_smartapi()

        socket = socket_factory(
            session["auth_token"],
            self._credentials.api_key,
            self._credentials.client_code,
            session["feed_token"],
        )
        socket.on_open = self._on_open
        socket.on_data = self._on_data
        socket.on_error = self._on_error
        socket.on_close = self._on_close
        self._socket = socket
        # ``connect`` blocks for the lifetime of the session in every published
        # version of the client, which is why the supervisor owns its own thread.
        socket.connect()

    def _token_list(self) -> list[dict[str, object]]:
        """Group configured tokens by exchange segment for subscription."""
        grouped: dict[int, list[str]] = {}
        for symbol in self._symbols:
            grouped.setdefault(symbol.exchange_type, []).append(symbol.token)
        return [
            {"exchangeType": exchange_type, "tokens": tokens}
            for exchange_type, tokens in sorted(grouped.items())
        ]

    def _on_open(self, _wsapp: Any = None) -> None:
        """Subscribe on connect and notify the engine of the discontinuity."""
        socket = self._socket
        if socket is None:  # pragma: no cover - defensive
            return
        config = self._adapter_config
        try:
            socket.subscribe(
                config.correlation_id, config.subscription_mode, self._token_list()
            )
        except Exception:
            _LOGGER.exception("subscription failed")
            return

        self._sessions += 1
        self._connected_event.set()
        _LOGGER.info(
            "subscribed %d token(s) in SnapQuote mode", len(self._symbols)
        )
        if self._gap_callback is not None:
            # Always notify, including on the first connection: it establishes the
            # invariant that engine statistics only ever start from a known-clean
            # state.
            self._gap_callback("websocket (re)connected")

    def _on_data(self, _wsapp: Any, message: Any = None) -> None:
        """Handle one callback payload.

        The signature is tolerant because published client versions call this with
        either ``(wsapp, message)`` or ``(message,)``.
        """
        received_ms = time.monotonic_ns() / 1_000_000.0
        payload = message if message is not None else _wsapp
        self._last_message_monotonic_ms = received_ms

        if is_heartbeat(payload):
            self._heartbeats += 1
            return
        try:
            records = coerce_payloads(payload)
        except PayloadError as exc:
            self._record_parse_error(str(exc))
            return

        for record in records:
            try:
                snapshot = parse_snapshot(
                    record,
                    received_monotonic_ms=received_ms,
                    symbol=self._symbol_name(record),
                )
            except PayloadError as exc:
                self._record_parse_error(str(exc))
                continue
            self._messages += 1
            self._queue.put(snapshot)

    def _symbol_name(self, record: Mapping[str, Any]) -> str:
        """Resolve the configured display name for a payload's token."""
        token = str(record.get("token", "") or record.get("tk", "")).strip()
        for symbol in self._symbols:
            if symbol.token == token:
                return symbol.symbol
        return token

    def _record_parse_error(self, detail: str) -> None:
        """Count and log a parse failure, throttling the log volume.

        Errors are never swallowed, but a malformed feed must not be able to
        generate millions of identical log lines.
        """
        self._parse_errors += 1
        if self._parse_errors <= 5 or self._parse_errors % 500 == 0:
            _LOGGER.error("payload parse error (%d total): %s", self._parse_errors, detail)

    def _on_error(self, _wsapp: Any, error: Any = None) -> None:
        """Log a socket error. The supervisor performs the actual reconnect."""
        self._connected_event.clear()
        _LOGGER.error("websocket error: %s", error if error is not None else _wsapp)

    def _on_close(self, _wsapp: Any = None, *args: Any) -> None:
        """Note a closed socket so the supervisor can reconnect."""
        self._connected_event.clear()
        _LOGGER.warning("websocket closed%s", f": {args}" if args else "")


def build_gap_notifier(
    notify: Callable[[str], None],
) -> Callable[[str], None]:
    """Wrap a gap callback so a failure inside it cannot kill the feed thread.

    The engine's reset path is straightforward, but it runs on the library's
    callback thread; an unhandled exception there would terminate the connection.
    """

    def _notify(reason: str) -> None:
        try:
            notify(reason)
        except Exception:
            _LOGGER.exception("gap callback failed")

    return _notify


def iterate_with_timeout(
    source: Iterable[RawSnapshot], stop: threading.Event
) -> Iterator[RawSnapshot]:
    """Yield from ``source`` until ``stop`` is set.

    A small helper so the runner's consumption loop reads declaratively and the
    stop condition is checked between every snapshot.
    """
    for snapshot in source:
        if stop.is_set():
            return
        yield snapshot


__all__ = [
    "AdapterUnavailableError",
    "AngelOneAdapter",
    "LoginError",
    "build_gap_notifier",
    "iterate_with_timeout",
]
