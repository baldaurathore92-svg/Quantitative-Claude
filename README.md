# Snapshot Quant Engine V4

A deterministic, snapshot-only quantitative engine for the NSE cash market, built
strictly on **Angel One SmartAPI V2 SnapQuote (subscription mode 3)**.

The engine core has **no third-party dependencies**. `smartapi-python` is needed
only for live mode and `rich` only for the fancier console; both are imported
lazily and both have complete fallbacks, so research, replay, tests and the
benchmark run on the standard library alone.

---

## What a snapshot feed cannot tell you

A retail snapshot feed publishes a periodic photograph of the top of the book. It
does **not** publish the order event stream. The following are therefore not
computable and are **not** attempted anywhere in this repository:

| Concept | Status | What this engine does instead |
|---|---|---|
| Aggressor side / aggressor ratio | impossible | not implemented at all |
| Exact iceberg detection | impossible | `refill` — a *documented proxy*: trade-confirmed consumption followed by replenishment at the same price |
| Exact spoof detection | impossible | not implemented; a quantity drop with no traded volume is simply not counted as consumption |
| Full per-level book imbalance | impossible beyond 5 levels | 5 published levels only: L1/L2 as the signal, L3–L5 and the exchange-published whole-book totals as *confidence* inputs |
| Queue position of your order | impossible | `queue_persistence` — survival of a *price level*, keyed on price |

Every approximation carries a docstring stating what it can and cannot detect.

---

## Architecture

```
RawSnapshot (adapter)
  → validator            structural checks, derived fields, gap flag
  → shared statistics    mid/spread/depth/stability, O(1) incremental
  → features             10 modules, declared dependency order
  → regime detector      deterministic, with hysteresis
  → quality gate         may block new signals without blocking state updates
  → confidence model     per-feature, from market conditions
  → composite scorer     normalised to [-1, +1]
  → adaptive threshold   clamped, volatility-driven
  → state machine        WARMUP → NEUTRAL → WATCH → LONG/SHORT → EXIT → COOLDOWN
  → EngineOutput         consumed by the renderer, the logger, or research code
```

Threading:

```
feed thread            engine thread                render thread
parse + enqueue   →    drain queue, process    →    every 1/fps: draw frame
(bounded queue,        (the measured path)          (never in the hot path)
 drop-oldest)
```

### Layout

```
snapshot_quant_v4/
  main.py            CLI: live | replay | synthetic
  runner.py          wiring, threads, latency tracking, shutdown
  config.py          typed, strictly validated configuration
  adapter/           angel_v2, parsing, queueing, replay, synthetic
  engine/            validator, stats, quality, confidence, regime,
                     composite, threshold, state_machine, execution,
                     quant_engine (SymbolEngine + EngineRegistry)
  features/          microprice, weighted_obi, depth_slope, spread,
                     spread_compression, queue_persistence, momentum,
                     acceleration, refill, ltp_confirmation
  buffers/           ring_buffer, rolling_mean, rolling_variance,
                     rolling_ema, monotonic_queue
  output/            views (presentation model), console (renderers)
  utils/             constants, types, clock, math_utils, logging_utils, totp
tests/               323 tests
bench/benchmark.py   per-snapshot latency measurement
```

---

## Quick start

No credentials or network required:

```bash
python -m snapshot_quant_v4.main --config config.example.json \
    --mode synthetic --count 2000 --paced-synthetic
```

Record a live session and replay it deterministically later:

```bash
python -m snapshot_quant_v4.main --config config.json --mode live \
    --record recordings/session.jsonl
python -m snapshot_quant_v4.main --config config.json --mode replay \
    --file recordings/session.jsonl --speed 0
```

Live mode needs the optional client and your credentials:

```bash
pip install 'smartapi-python>=1.4' 'websocket-client>=1.6' 'rich>=13'
export SQ4_API_KEY=... SQ4_CLIENT_CODE=... SQ4_PIN=... SQ4_TOTP_SECRET=...
```

`SQ4_TOTP_SECRET` is the **base32 seed**, not a six-digit code. Codes are
generated in-process (RFC 6238, standard library only) and every credential is
registered with the logging redactor before anything else starts.

---

## Design decisions that matter

**Everything is in ticks, not rupees or basis points.** A price difference is
meaningless across instruments until it is divided by the tick size. The
order-book imbalance weight is `exp(-distance_in_ticks / decay_ticks)`; a
rupee-denominated decay of the form `mid * 0.005` evaluates to roughly 250 ticks
on a 2500 rupee instrument, which makes every weight ≈ 1 and silently turns a
weighted imbalance into a plain sum.

**Comparisons between snapshots are keyed on price, never on ladder index.**
When the best bid improves by a tick, `bids[0]` refers to a different price. An
index-based comparison reports a size collapse that never happened. Both
`queue_persistence` and `refill` look up the previous price with
`Snapshot.level_at`, and there is a regression test for exactly this.

**EMAs decay in time, not in samples.** A snapshot feed is event driven; a fixed
per-sample `alpha` represents half a second in a busy market and thirty seconds
in a quiet one, so the derived feature is not stationary. All EMAs use
`alpha = 1 - exp(-ln2 · Δt / half_life)` with `Δt` from the exchange timestamp.

**Consumption is distinguished from cancellation using traded volume.** A
quantity drop at the touch either traded away (aggressive flow, and a refill
afterwards is real support) or was cancelled (liquidity fading). The only
snapshot-observable discriminator is the change in cumulative day volume, so
`refill` requires the drop to be volume-confirmed. Growth without a preceding,
confirmed consumption never triggers anything.

**The composite is normalised.** `score = Σ(value·weight·confidence) /
Σ(weight·confidence)`, which keeps it in `[-1, +1]` and comparable across
regimes. A corollary that is easy to get wrong: because the score is invariant to
uniform weight scaling, "reduce every weight in a noisy regime" does *nothing*.
Reduced trust in noise is therefore expressed through a regime confidence
multiplier and a higher threshold, where it has the intended effect.

**Microprice and L1 imbalance are the same quantity.** Algebraically,
`(microprice − mid) / (spread/2) ≡ (bid_qty − ask_qty) / (bid_qty + ask_qty)`.
Giving both independent weight would count one observation twice, so the
composite caps their *joint* share of the weight mass
(`composite.collinear_groups`).

**Unsigned features never enter a signed score.** Spread and spread compression
have magnitude but no direction, so they are typed `QUALITY` and routed into the
confidence model. Feeding them into the score would be a category error.

**Invalid features are excluded, not zeroed.** Zero asserts "this feature says
neutral"; the truth is "this feature has nothing to say yet".

**A feed gap invalidates every incremental statistic that spans it.** On a gap,
a backwards timestamp, a volume regression or a reconnect, the engine closes any
open position at the current book, resets every estimator, and returns to
`WARMUP`. Continuing to feed EMAs and sliding variances across a discontinuity is
silently wrong and nearly impossible to spot in live output.

**The threshold is clamped.** Every dispersion term tends to zero in a quiet
book, so an unclamped adaptive threshold converges towards zero exactly when the
market is least informative. Until the dispersion estimators are ready the
threshold is pinned to its maximum: the engine refuses to trade rather than
guessing.

**Execution and costs are separate from the signal path.** No feature, score,
regime or threshold imports `engine/execution.py`. Entry aggression expressed in
basis points is converted to ticks against the live price and rounded onto the
grid, is never better than the touch, and — with `use_depth_walk` — walks the
visible ladder and reports partial fills. The optional `enforce_cost_gate`
compares the configured target against the modelled round-trip cost; it is off by
default so signal behaviour is never implicitly coupled to a cost assumption.

**Stops are evaluated on the exit side of the book**, not on the mid.

---

## Performance

Measured on this machine, CPython 3.11, single thread, 20 000 synthetic
snapshots per run, three runs:

```
1 symbol(s) run 1      n=20000   mean=  275.2us  p50=  272.1us  p90=  283.2us  p99=  327.9us  max=   532.6us   3627 snapshots/s
1 symbol(s) run 2      n=20000   mean=  273.4us  p50=  270.8us  p90=  280.5us  p99=  323.2us  max=   499.8us   3651 snapshots/s
1 symbol(s) run 3      n=20000   mean=  274.5us  p50=  271.4us  p90=  282.2us  p99=  331.7us  max=   546.9us   3636 snapshots/s
```

Reproduce with `python bench/benchmark.py --count 20000 --repeat 3`.

The measured region is `SymbolEngine.process`: validation, statistics, all ten
features, regime, quality, confidence, composite, threshold and the state
machine. It excludes the websocket, terminal rendering, logging and the
construction of the human-readable reason strings, all of which are either on
other threads or built after the measurement ends.

Complexity is O(1) per snapshot with respect to history: no buffer is rescanned,
no list is sliced, no copy is taken. The drift-corrected rolling sums perform one
exact recomputation every `recompute_interval` pushes, which is what the p99 tail
above reflects. Rolling extrema use a monotonic deque (amortised O(1)).

---

## Development

```bash
python -m pytest -q            # 323 tests
ruff check .                   # lint
mypy                           # strict
python bench/benchmark.py      # latency
```

Every feature is a pure transformation of snapshots plus its own resettable
estimators, with no access to the network or to the state machine, so a feature
test builds two books by hand and asserts on the result. Time is injected through
a `Clock` protocol, so state-machine timeouts are tested with a `ManualClock`
rather than with sleeps. The live adapter is tested against injected fakes for
`SmartConnect` and `SmartWebSocketV2`, covering login retries, subscription,
payload shapes and the reconnect loop without a network.

---

## Configuration

Copy `config.example.json` and edit. Loading is strict: an unknown key is an
error, not a silent no-op, and every section validates its own ranges at
construction. All mapping fields are wrapped read-only, so no component can
mutate shared configuration at runtime.

Notable sections: `validation` (hard structural limits), `quality` (soft signal
gating), `features` (one block per feature), `confidence` (per-feature
sensitivity exponents), `regime`, `composite` (per-regime weight tables),
`threshold`, `state_machine`, `execution` (+ `cost`), `runtime`, `adapter`.

---

## Scope

This is research and signal infrastructure. It models fills and costs; it does
**not** place orders. Nothing here is investment advice, and the synthetic
generator is a test fixture — no result derived from it says anything about a
strategy's profitability.
