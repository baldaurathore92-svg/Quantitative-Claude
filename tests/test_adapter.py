"""Adapter tests: payload parsing, queueing, replay and connection lifecycle.

The live adapter is exercised with injected fakes for ``SmartConnect`` and
``SmartWebSocketV2``, so the login, subscription, reconnection and payload paths
are all covered without a network and without the optional third-party package.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest
from snapshot_quant_v4.adapter.angel_v2 import (
    AdapterUnavailableError,
    AngelOneAdapter,
    LoginError,
    _import_smartapi,
    build_gap_notifier,
)
from snapshot_quant_v4.adapter.parsing import (
    PayloadError,
    coerce_payloads,
    is_heartbeat,
    parse_depth,
    parse_snapshot,
    price_divisor,
    snapshot_from_json,
    snapshot_to_json,
)
from snapshot_quant_v4.adapter.queueing import SnapshotQueue
from snapshot_quant_v4.adapter.replay import ReplaySource, SnapshotRecorder
from snapshot_quant_v4.adapter.synthetic import SyntheticConfig, SyntheticSource
from snapshot_quant_v4.utils.constants import ExchangeType

from conftest import TOKEN, build_config, make_snapshot

SNAPQUOTE_PAYLOAD: dict[str, Any] = {
    "token": TOKEN,
    "exchange_type": 1,
    "exchange_timestamp": 1_753_500_000_123,
    "sequence_number": 42,
    "last_traded_price": 80_005,
    "last_traded_quantity": 30,
    "average_traded_price": 80_002,
    "volume_trade_for_the_day": 1_234_567,
    "total_buy_quantity": 90_000.0,
    "total_sell_quantity": 85_000.0,
    "open_price_of_the_day": 79_500,
    "high_price_of_the_day": 80_500,
    "low_price_of_the_day": 79_000,
    "closed_price": 79_800,
    "best_5_buy_data": [
        {"price": 79_995, "quantity": 600, "no of orders": 4},
        {"price": 79_990, "quantity": 450, "no of orders": 3},
    ],
    "best_5_sell_data": [
        {"price": 80_005, "quantity": 500, "no of orders": 2},
        {"price": 80_010, "quantity": 400, "no of orders": 5},
    ],
}


class TestHeartbeatDetection:
    @pytest.mark.parametrize("payload", ["pong", "ping", "", b'"pong"', None, "ok"])
    def test_keep_alives_are_recognised(self, payload: Any) -> None:
        assert is_heartbeat(payload)

    def test_market_data_is_not_a_heartbeat(self) -> None:
        assert not is_heartbeat(SNAPQUOTE_PAYLOAD)
        assert not is_heartbeat(json.dumps(SNAPQUOTE_PAYLOAD))

    def test_mapping_without_a_token_is_treated_as_a_control_frame(self) -> None:
        assert is_heartbeat({"status": "subscribed"})


class TestPayloadCoercion:
    def test_mapping_passes_through(self) -> None:
        assert coerce_payloads(SNAPQUOTE_PAYLOAD) == [SNAPQUOTE_PAYLOAD]

    def test_json_string_is_parsed(self) -> None:
        assert coerce_payloads(json.dumps(SNAPQUOTE_PAYLOAD))[0]["token"] == TOKEN

    def test_json_bytes_are_parsed(self) -> None:
        raw = json.dumps(SNAPQUOTE_PAYLOAD).encode("utf-8")
        assert coerce_payloads(raw)[0]["token"] == TOKEN

    def test_list_is_flattened(self) -> None:
        batch = [SNAPQUOTE_PAYLOAD, json.dumps(SNAPQUOTE_PAYLOAD)]
        assert len(coerce_payloads(batch)) == 2

    def test_heartbeat_yields_nothing(self) -> None:
        assert coerce_payloads("pong") == []

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(PayloadError):
            coerce_payloads('{"token": ')

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(PayloadError):
            coerce_payloads(3.14)

    def test_invalid_utf8_raises(self) -> None:
        with pytest.raises(PayloadError):
            coerce_payloads(b"\xff\xfe{")


class TestSnapshotParsing:
    def test_parses_prices_quantities_and_order_counts(self) -> None:
        snapshot = parse_snapshot(
            SNAPQUOTE_PAYLOAD, received_monotonic_ms=1.0, symbol="SBIN"
        )
        assert snapshot.token == TOKEN
        assert snapshot.symbol == "SBIN"
        assert snapshot.bids[0].price == pytest.approx(799.95)
        assert snapshot.bids[0].price_paise == 79_995
        assert snapshot.bids[0].orders == 4
        assert snapshot.asks[0].price == pytest.approx(800.05)
        assert snapshot.last_traded_price == pytest.approx(800.05)
        assert snapshot.volume_traded_today == 1_234_567
        assert snapshot.exchange_timestamp_ms == 1_753_500_000_123

    def test_alternative_field_spellings_are_accepted(self) -> None:
        payload = {
            "tk": TOKEN,
            "e": 1,
            "ft": 1_753_500_000_000,
            "lp": 80_000,
            "v": 99,
            "bids": [{"p": 79_995, "q": 600, "orders": 2}],
            "asks": [{"p": 80_005, "q": 500, "orders": 2}],
        }
        snapshot = parse_snapshot(payload, received_monotonic_ms=0.0)
        assert snapshot.bids[0].quantity == 600
        assert snapshot.asks[0].price == pytest.approx(800.05)

    def test_missing_order_count_defaults_to_one(self) -> None:
        # Zero orders with positive quantity is contradictory and several ratios
        # divide by the count.
        levels = parse_depth(
            [{"price": 79_995, "quantity": 600}], 100, descending=True
        )
        assert levels[0].orders == 1

    def test_padding_levels_are_dropped(self) -> None:
        levels = parse_depth(
            [
                {"price": 79_995, "quantity": 600, "no of orders": 2},
                {"price": 0, "quantity": 0, "no of orders": 0},
            ],
            100,
            descending=True,
        )
        assert len(levels) == 1

    def test_depth_is_sorted_regardless_of_wire_order(self) -> None:
        levels = parse_depth(
            [
                {"price": 79_990, "quantity": 100, "no of orders": 1},
                {"price": 79_995, "quantity": 200, "no of orders": 1},
            ],
            100,
            descending=True,
        )
        assert [entry.price_paise for entry in levels] == [79_995, 79_990]

    def test_depth_is_capped_at_five_levels(self) -> None:
        entries = [
            {"price": 80_000 - index * 5, "quantity": 100, "no of orders": 1}
            for index in range(9)
        ]
        assert len(parse_depth(entries, 100, descending=True)) == 5

    def test_missing_token_raises(self) -> None:
        with pytest.raises(PayloadError):
            parse_snapshot({"best_5_buy_data": []}, received_monotonic_ms=0.0)

    def test_no_depth_raises_with_a_mode_hint(self) -> None:
        with pytest.raises(PayloadError) as info:
            parse_snapshot({"token": TOKEN}, received_monotonic_ms=0.0)
        assert "mode 3" in str(info.value)

    def test_currency_segment_uses_its_own_divisor(self) -> None:
        assert price_divisor(int(ExchangeType.CDE_FO)) == 10_000_000
        assert price_divisor(int(ExchangeType.NSE_CM)) == 100
        assert price_divisor(999) == 100

    def test_currency_prices_are_converted_and_keyed_in_paise(self) -> None:
        payload = {
            "token": "1",
            "exchange_type": int(ExchangeType.CDE_FO),
            "best_5_buy_data": [{"price": 830_000_000, "quantity": 10, "no of orders": 1}],
            "best_5_sell_data": [{"price": 830_500_000, "quantity": 10, "no of orders": 1}],
        }
        snapshot = parse_snapshot(payload, received_monotonic_ms=0.0)
        assert snapshot.bids[0].price == pytest.approx(83.0)
        assert snapshot.bids[0].price_paise == 8_300


class TestJsonRoundTrip:
    def test_round_trip_preserves_the_book(self) -> None:
        original = make_snapshot()
        record = json.loads(snapshot_to_json(original))
        restored = snapshot_from_json(record, received_monotonic_ms=5.0)
        assert restored.token == original.token
        assert restored.bids == original.bids
        assert restored.asks == original.asks
        assert restored.volume_traded_today == original.volume_traded_today
        assert restored.exchange_timestamp_ms == original.exchange_timestamp_ms

    def test_record_without_a_token_raises(self) -> None:
        with pytest.raises(PayloadError):
            snapshot_from_json({"bids": []}, received_monotonic_ms=0.0)


class TestSnapshotQueue:
    def test_drops_the_oldest_when_full(self) -> None:
        queue = SnapshotQueue(2)
        first = make_snapshot(sequence=1)
        second = make_snapshot(sequence=2)
        third = make_snapshot(sequence=3)
        for snapshot in (first, second, third):
            queue.put(snapshot)
        queue.close()
        drained = list(queue.drain(timeout=0.05))
        assert queue.dropped == 1
        # The newest snapshot survives; the oldest was discarded.
        assert [snapshot.sequence_number for snapshot in drained] == [2, 3]

    def test_close_releases_a_blocked_consumer(self) -> None:
        queue = SnapshotQueue(4)
        received: list[int] = []

        def consume() -> None:
            received.extend(
                snapshot.sequence_number for snapshot in queue.drain(timeout=0.05)
            )

        thread = threading.Thread(target=consume)
        thread.start()
        queue.put(make_snapshot(sequence=7))
        time.sleep(0.05)
        queue.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert received == [7]

    def test_rejects_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            SnapshotQueue(0)


class TestReplayAndRecorder:
    def test_recorder_then_replay_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        snapshots = [
            make_snapshot(sequence=index, timestamp_ms=1_000 + index)
            for index in range(5)
        ]
        with SnapshotRecorder(path) as recorder:
            for snapshot in snapshots:
                recorder.write(snapshot)
        assert recorder.written == 5

        source = ReplaySource(path, strict=True)
        source.start()
        replayed = list(source.snapshots())
        assert [snapshot.sequence_number for snapshot in replayed] == [0, 1, 2, 3, 4]
        assert replayed[0].bids == snapshots[0].bids

    def test_max_snapshots_limits_playback(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        with SnapshotRecorder(path) as recorder:
            for index in range(10):
                recorder.write(make_snapshot(sequence=index))
        source = ReplaySource(path, max_snapshots=3)
        source.start()
        assert len(list(source.snapshots())) == 3

    def test_malformed_lines_are_skipped_when_not_strict(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        good = snapshot_to_json(make_snapshot(sequence=1))
        path.write_text(f"{good}\nnot json\n\n{good}\n", encoding="utf-8")
        source = ReplaySource(path, strict=False)
        source.start()
        assert len(list(source.snapshots())) == 2
        assert source.errors == 1

    def test_malformed_lines_raise_in_strict_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        source = ReplaySource(path, strict=True)
        source.start()
        with pytest.raises(PayloadError):
            list(source.snapshots())

    def test_missing_file_is_reported_at_start(self, tmp_path: Path) -> None:
        source = ReplaySource(tmp_path / "absent.jsonl")
        with pytest.raises(FileNotFoundError):
            source.start()

    def test_stop_ends_playback(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        with SnapshotRecorder(path) as recorder:
            for index in range(20):
                recorder.write(make_snapshot(sequence=index))
        source = ReplaySource(path)
        source.start()
        collected = []
        for snapshot in source.snapshots():
            collected.append(snapshot)
            if len(collected) == 3:
                source.stop()
        assert len(collected) == 3

    def test_recorder_requires_open(self, tmp_path: Path) -> None:
        recorder = SnapshotRecorder(tmp_path / "x.jsonl")
        with pytest.raises(RuntimeError):
            recorder.write(make_snapshot())

    def test_rejects_negative_speed(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            ReplaySource(tmp_path / "x.jsonl", speed=-1.0)


class TestSyntheticSource:
    def test_is_deterministic_for_a_seed(self) -> None:
        config = SyntheticConfig(count=50, seed=1234)
        first = SyntheticSource(config)
        first.start()
        second = SyntheticSource(config)
        second.start()
        left = [snapshot.bids[0].quantity for snapshot in first.snapshots()]
        right = [snapshot.bids[0].quantity for snapshot in second.snapshots()]
        assert left == right

    def test_volume_only_increases_and_book_is_uncrossed(self) -> None:
        source = SyntheticSource(SyntheticConfig(count=200))
        source.start()
        previous_volume = -1
        previous_timestamp = -1
        for snapshot in source.snapshots():
            assert snapshot.volume_traded_today >= previous_volume
            assert snapshot.exchange_timestamp_ms > previous_timestamp
            assert snapshot.asks[0].price_paise > snapshot.bids[0].price_paise
            assert all(level.orders >= 1 for level in snapshot.bids)
            previous_volume = snapshot.volume_traded_today
            previous_timestamp = snapshot.exchange_timestamp_ms

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            SyntheticConfig(count=0)
        with pytest.raises(ValueError):
            SyntheticConfig(depth_decay=1.5)


class FakeSmartConnect:
    """Stand-in for ``SmartConnect``."""

    def __init__(self, api_key: str = "", fail_times: int = 0) -> None:
        self.api_key = api_key
        self.fail_times = fail_times
        self.calls: list[tuple[str, str, str]] = []


    def generateSession(self, client_code: str, pin: str, totp: str) -> dict:
        self.calls.append((client_code, pin, totp))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient login failure")
        return {
            "status": True,
            "data": {
                "jwtToken": "jwt-token",
                "refreshToken": "refresh-token",
                "feedToken": "feed-token",
            },
        }


class FakeWebSocket:
    """Stand-in for ``SmartWebSocketV2``.

    ``connect`` invokes ``on_open`` and then blocks until released, mirroring the
    real client's blocking behaviour so the adapter's threading is exercised.
    """

    instances: ClassVar[list[FakeWebSocket]] = []

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str) -> None:
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.subscriptions: list[tuple[str, int, list]] = []
        self.on_open = None
        self.on_data = None
        self.on_error = None
        self.on_close = None
        self._release = threading.Event()
        FakeWebSocket.instances.append(self)

    def connect(self) -> None:
        if self.on_open is not None:
            self.on_open(self)
        self._release.wait(timeout=5.0)

    def subscribe(self, correlation_id: str, mode: int, token_list: list) -> None:
        self.subscriptions.append((correlation_id, mode, token_list))

    def close_connection(self) -> None:
        self._release.set()

    def push(self, payload: Any) -> None:
        """Deliver a payload through the data callback."""
        assert self.on_data is not None
        self.on_data(self, payload)


@pytest.fixture(autouse=True)
def _clear_fake_sockets() -> None:
    FakeWebSocket.instances.clear()


def build_adapter(**kwargs: Any) -> AngelOneAdapter:
    """Build an adapter wired to the fakes."""
    config = build_config(
        credentials={
            "api_key": "test-key",
            "client_code": "AB1234",
            "pin": "1234",
            "totp_secret": "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ",
        },
        adapter={"reconnect_initial_ms": 10.0, "reconnect_max_ms": 20.0},
    )
    kwargs.setdefault(
        "smart_connect_factory", lambda **factory_kwargs: FakeSmartConnect(**factory_kwargs)
    )
    kwargs.setdefault("socket_factory", FakeWebSocket)
    return AngelOneAdapter(config, **kwargs)


class TestAngelOneAdapter:
    def test_subscribes_in_snapquote_mode_and_notifies_the_gap_callback(self) -> None:
        gaps: list[str] = []
        adapter = build_adapter(gap_callback=build_gap_notifier(gaps.append))
        adapter.start()
        deadline = time.monotonic() + 3.0
        while not FakeWebSocket.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        socket = FakeWebSocket.instances[0]
        while not socket.subscriptions and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            correlation, mode, tokens = socket.subscriptions[0]
            assert mode == 3
            assert tokens == [{"exchangeType": 1, "tokens": [TOKEN]}]
            assert correlation == "sq4"
            assert gaps and "connected" in gaps[0]
        finally:
            adapter.stop()

    def test_payloads_reach_the_queue_and_heartbeats_do_not(self) -> None:
        adapter = build_adapter()
        adapter.start()
        deadline = time.monotonic() + 3.0
        while not FakeWebSocket.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        socket = FakeWebSocket.instances[0]
        try:
            socket.push("pong")
            socket.push(SNAPQUOTE_PAYLOAD)
            socket.push(json.dumps(SNAPQUOTE_PAYLOAD))
            socket.push("{bad json")
            time.sleep(0.05)
            adapter.stop()
            received = list(adapter.snapshots())
            assert len(received) == 2
            assert adapter.heartbeats == 1
            assert adapter.parse_errors == 1
            assert received[0].symbol == "SBIN"
        finally:
            adapter.stop()

    def test_login_retries_before_succeeding(self) -> None:
        connect = FakeSmartConnect(fail_times=1)
        adapter = build_adapter(smart_connect_factory=lambda **_: connect)
        adapter.start()
        deadline = time.monotonic() + 5.0
        while not FakeWebSocket.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            assert len(connect.calls) >= 2
        finally:
            adapter.stop()

    def test_login_failure_is_reported(self) -> None:
        class AlwaysFails:
            def generateSession(self, *_: Any) -> dict:
                raise RuntimeError("bad credentials")

        adapter = build_adapter(
            smart_connect_factory=lambda **_: AlwaysFails(),
            queue_size=8,
        )
        with pytest.raises(LoginError):
            adapter._login()

    def test_session_response_without_tokens_is_rejected(self) -> None:
        with pytest.raises(LoginError):
            AngelOneAdapter._extract_tokens(object(), {"status": True, "data": {}})
        with pytest.raises(LoginError):
            AngelOneAdapter._extract_tokens(object(), {"status": False, "message": "no"})
        with pytest.raises(LoginError):
            AngelOneAdapter._extract_tokens(object(), "unexpected")

    def test_feed_token_falls_back_to_the_client_getter(self) -> None:
        class WithGetter:
            def getfeedToken(self) -> str:
                return "from-getter"

        tokens = AngelOneAdapter._extract_tokens(
            WithGetter(), {"status": True, "data": {"jwtToken": "jwt"}}
        )
        assert tokens["feed_token"] == "from-getter"

    def test_reconnect_loop_creates_a_new_session(self) -> None:
        gaps: list[str] = []
        adapter = build_adapter(gap_callback=build_gap_notifier(gaps.append))
        adapter.start()
        deadline = time.monotonic() + 5.0
        while not FakeWebSocket.instances and time.monotonic() < deadline:
            time.sleep(0.01)
        first = FakeWebSocket.instances[0]
        # Simulate a server-side close: the supervisor must loop and reconnect
        # rather than recurse.
        first.close_connection()
        while len(FakeWebSocket.instances) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            assert len(FakeWebSocket.instances) >= 2
            assert adapter.reconnects >= 1
            assert len(gaps) >= 2
        finally:
            adapter.stop()

    def test_double_start_is_rejected(self) -> None:
        adapter = build_adapter()
        adapter.start()
        try:
            with pytest.raises(RuntimeError):
                adapter.start()
        finally:
            adapter.stop()

    def test_requires_at_least_one_symbol(self) -> None:
        with pytest.raises(ValueError):
            AngelOneAdapter(build_config(symbols=[]))

    def test_gap_notifier_swallows_callback_failures(self) -> None:
        def explode(_reason: str) -> None:
            raise RuntimeError("callback defect")

        # A failure in the engine's reset path must not kill the feed thread.
        build_gap_notifier(explode)("reason")

    def test_missing_library_raises_with_installation_guidance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_: Any, **__: Any) -> Any:
            raise ImportError("no module named SmartApi")

        monkeypatch.setattr("builtins.__import__", fail)
        with pytest.raises(AdapterUnavailableError) as info:
            _import_smartapi()
        assert "smartapi-python" in str(info.value)
