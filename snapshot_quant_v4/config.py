"""Typed, strictly validated configuration.

Design rules
------------
1.  **Every tunable lives here.** No feature, no threshold and no timeout is
    written as a literal inside the engine. That is what makes the engine
    re-parameterisable for research without editing code.
2.  **Strict loading.** An unknown key in ``config.json`` is an error, not a
    silent no-op. A typo in a parameter name would otherwise be invisible and
    would leave the engine running with a default the operator did not intend.
3.  **Range validation at construction.** Each section validates its own
    invariants in ``__post_init__`` and raises :class:`ConfigError`. The engine
    can therefore assume its configuration is sane and does not re-check
    parameters on the hot path.
4.  **Immutability.** Sections are frozen dataclasses and mapping fields are
    wrapped in ``MappingProxyType``, so no component can mutate shared
    configuration at runtime. This is what "no global mutable state" requires
    in practice.
5.  **Secrets are never defaulted from the file alone.** Credentials may be
    supplied by environment variables, which is the safer deployment path, and
    are registered with the logging redactor by the runner.
"""

from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from collections import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, get_args, get_origin

from .utils.constants import (
    DEFAULT_TICK_SIZE,
    F_ACCELERATION,
    F_DEPTH_SLOPE,
    F_LTP_CONFIRMATION,
    F_MICROPRICE,
    F_MOMENTUM,
    F_QUEUE_PERSISTENCE,
    F_REFILL,
    F_WEIGHTED_OBI,
    ExchangeType,
    SubscriptionMode,
)

__all__ = [
    "AccelerationConfig",
    "AdapterConfig",
    "AppConfig",
    "CompositeConfig",
    "ConfidenceConfig",
    "ConfigError",
    "CostConfig",
    "CredentialsConfig",
    "DepthSlopeConfig",
    "ExecutionConfig",
    "FeatureSensitivity",
    "FeaturesConfig",
    "LtpConfirmationConfig",
    "MicropriceConfig",
    "MomentumConfig",
    "QualityConfig",
    "QueuePersistenceConfig",
    "RefillConfig",
    "RegimeConfig",
    "RuntimeConfig",
    "SpreadCompressionConfig",
    "SpreadConfig",
    "StateMachineConfig",
    "SymbolConfig",
    "ThresholdConfig",
    "ValidationConfig",
    "WeightedObiConfig",
    "config_from_mapping",
    "load_config",
]


class ConfigError(ValueError):
    """Raised for any malformed, unknown or out-of-range configuration value."""


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _require_positive(name: str, value: float) -> None:
    if not value > 0.0:
        raise ConfigError(f"{name} must be > 0, got {value}")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ConfigError(f"{name} must be >= 0, got {value}")


def _require_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ConfigError(f"{name} must be in [{low}, {high}], got {value}")


def _require_ordered(low_name: str, low: float, high_name: str, high: float) -> None:
    if not low < high:
        raise ConfigError(f"{low_name} ({low}) must be < {high_name} ({high})")


def _freeze_mapping(instance: object, attribute: str, mapping: Mapping[str, Any]) -> None:
    """Replace a mapping attribute of a frozen dataclass with a read-only view."""
    object.__setattr__(instance, attribute, MappingProxyType(dict(mapping)))


# --------------------------------------------------------------------------- #
# Instruments and credentials
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SymbolConfig:
    """One subscribed instrument.

    ``tick_size`` is per symbol rather than global: the engine expresses every
    price difference in ticks, so an incorrect tick size silently rescales every
    feature for that instrument.
    """

    token: str
    symbol: str = ""
    exchange_type: int = int(ExchangeType.NSE_CM)
    tick_size: float = DEFAULT_TICK_SIZE
    quantity: int = 1

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ConfigError("symbol.token must be a non-empty string")
        _require_positive(f"symbol[{self.token}].tick_size", self.tick_size)
        if self.quantity <= 0:
            raise ConfigError(f"symbol[{self.token}].quantity must be > 0")
        try:
            ExchangeType(self.exchange_type)
        except ValueError as exc:
            raise ConfigError(
                f"symbol[{self.token}].exchange_type {self.exchange_type} is not a "
                "known SmartAPI exchange code"
            ) from exc
        if not self.symbol:
            object.__setattr__(self, "symbol", self.token)


@dataclass(frozen=True, slots=True)
class CredentialsConfig:
    """Angel One SmartAPI credentials.

    ``totp_secret`` is the base32 seed, not a generated code: a code is valid
    for one 30 second step and cannot be persisted usefully. Values may be
    supplied through the environment (see :func:`load_config`), which keeps them
    out of the repository.
    """

    api_key: str = ""
    client_code: str = ""
    pin: str = ""
    totp_secret: str = ""

    def secret_values(self) -> tuple[str, ...]:
        """Return every value that must be redacted from logs."""
        return tuple(v for v in (self.api_key, self.pin, self.totp_secret) if v)

    def validate_for_live(self) -> None:
        """Ensure all fields required for a live session are present.

        Deliberately *not* called from ``__post_init__``: replay and unit tests
        legitimately run without credentials.
        """
        missing = [
            name
            for name, value in (
                ("api_key", self.api_key),
                ("client_code", self.client_code),
                ("pin", self.pin),
                ("totp_secret", self.totp_secret),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "live mode requires credentials: missing " + ", ".join(missing)
            )


# --------------------------------------------------------------------------- #
# Validation and quality gating
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Hard structural checks. A failing snapshot is discarded entirely."""

    max_spread_ticks: float = 25.0
    max_snapshot_gap_ms: float = 3_000.0
    max_staleness_ms: float = 8_000.0
    max_price_gap_ticks: float = 40.0
    min_price: float = 0.01
    require_tick_alignment: bool = False
    allow_locked_book: bool = False
    reject_duplicates: bool = True
    require_ltp: bool = True

    def __post_init__(self) -> None:
        _require_positive("validation.max_spread_ticks", self.max_spread_ticks)
        _require_positive("validation.max_snapshot_gap_ms", self.max_snapshot_gap_ms)
        _require_positive("validation.max_staleness_ms", self.max_staleness_ms)
        _require_positive("validation.max_price_gap_ticks", self.max_price_gap_ticks)
        _require_positive("validation.min_price", self.min_price)


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Soft gating. The snapshot still updates state; new signals are blocked.

    ``min_relative_depth`` compares the current top-of-book depth against this
    symbol's own recent depth (a rolling geometric mean). An absolute quantity
    threshold cannot work across instruments whose typical depth differs by
    three orders of magnitude.
    """

    max_signal_spread_ticks: float = 4.0
    min_depth_levels: int = 2
    min_top_quantity: int = 0
    min_relative_depth: float = 0.35
    min_orders_per_level: int = 1
    gap_block_ms: float = 2_000.0
    warmup_snapshots: int = 60

    def __post_init__(self) -> None:
        _require_positive("quality.max_signal_spread_ticks", self.max_signal_spread_ticks)
        if not 1 <= self.min_depth_levels <= 5:
            raise ConfigError("quality.min_depth_levels must be in 1..5")
        _require_non_negative("quality.min_top_quantity", self.min_top_quantity)
        _require_range("quality.min_relative_depth", self.min_relative_depth, 0.0, 1.0)
        _require_non_negative("quality.min_orders_per_level", self.min_orders_per_level)
        _require_non_negative("quality.gap_block_ms", self.gap_block_ms)
        if self.warmup_snapshots < 1:
            raise ConfigError("quality.warmup_snapshots must be >= 1")


# --------------------------------------------------------------------------- #
# Feature parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MicropriceConfig:
    """Microprice tilt parameters.

    ``smooth_half_life_ms`` lightly smooths the tilt. The unsmoothed tilt is
    still reported as the raw value so that research code sees the untouched
    observation.
    """

    smooth_half_life_ms: float = 200.0
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        _require_positive("features.microprice.smooth_half_life_ms", self.smooth_half_life_ms)
        _require_positive("features.microprice.max_dt_ms", self.max_dt_ms)


@dataclass(frozen=True, slots=True)
class WeightedObiConfig:
    """Weighted order-book imbalance parameters.

    ``decay_ticks`` is the decay constant of ``exp(-distance_in_ticks / decay)``.
    Measuring distance in ticks rather than rupees is what makes the weight
    scale-free: a rupee-denominated decay constant produces near-identical
    weights for every level of a tight book and completely different behaviour
    on a 150 rupee versus a 3000 rupee instrument.

    ``primary_levels`` bounds the directional signal (L1/L2 by default). The
    deeper levels and the exchange-published whole-book totals are used only to
    modulate confidence via ``deep_agreement_weight`` and
    ``aggregate_agreement_weight``.
    """

    decay_ticks: float = 2.5
    primary_levels: int = 2
    deep_levels: int = 5
    deep_agreement_weight: float = 0.35
    aggregate_agreement_weight: float = 0.20
    aggregate_max_ratio: float = 12.0

    def __post_init__(self) -> None:
        _require_positive("features.weighted_obi.decay_ticks", self.decay_ticks)
        if not 1 <= self.primary_levels <= 5:
            raise ConfigError("features.weighted_obi.primary_levels must be in 1..5")
        if not self.primary_levels <= self.deep_levels <= 5:
            raise ConfigError(
                "features.weighted_obi.deep_levels must be in primary_levels..5"
            )
        _require_range(
            "features.weighted_obi.deep_agreement_weight",
            self.deep_agreement_weight,
            0.0,
            1.0,
        )
        _require_range(
            "features.weighted_obi.aggregate_agreement_weight",
            self.aggregate_agreement_weight,
            0.0,
            1.0,
        )
        _require_positive(
            "features.weighted_obi.aggregate_max_ratio", self.aggregate_max_ratio
        )


@dataclass(frozen=True, slots=True)
class DepthSlopeConfig:
    """Relative depth decay parameters."""

    scale: float = 0.9
    use_deep_levels: bool = True
    deep_levels: int = 5

    def __post_init__(self) -> None:
        _require_positive("features.depth_slope.scale", self.scale)
        if not 3 <= self.deep_levels <= 5:
            raise ConfigError("features.depth_slope.deep_levels must be in 3..5")


@dataclass(frozen=True, slots=True)
class SpreadConfig:
    """Spread quality mapping: ``tight_ticks`` scores 1.0, ``wide_ticks`` 0.0."""

    tight_ticks: float = 1.0
    wide_ticks: float = 6.0
    variance_window: int = 120

    def __post_init__(self) -> None:
        _require_positive("features.spread.tight_ticks", self.tight_ticks)
        _require_ordered(
            "features.spread.tight_ticks",
            self.tight_ticks,
            "features.spread.wide_ticks",
            self.wide_ticks,
        )
        if self.variance_window < 2:
            raise ConfigError("features.spread.variance_window must be >= 2")


@dataclass(frozen=True, slots=True)
class SpreadCompressionConfig:
    """Spread compression parameters (fast versus slow spread EMA)."""

    fast_half_life_ms: float = 500.0
    slow_half_life_ms: float = 4_000.0
    scale_ticks: float = 0.5
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        _require_positive(
            "features.spread_compression.fast_half_life_ms", self.fast_half_life_ms
        )
        _require_ordered(
            "features.spread_compression.fast_half_life_ms",
            self.fast_half_life_ms,
            "features.spread_compression.slow_half_life_ms",
            self.slow_half_life_ms,
        )
        _require_positive("features.spread_compression.scale_ticks", self.scale_ticks)
        _require_positive("features.spread_compression.max_dt_ms", self.max_dt_ms)


@dataclass(frozen=True, slots=True)
class QueuePersistenceConfig:
    """Queue persistence parameters.

    ``tolerance`` is the fraction of quantity a level may lose while still
    counting as "persisted"; ``window`` is the number of snapshots over which
    the survival rate is averaged.
    """

    window: int = 40
    tolerance: float = 0.15
    min_samples: int = 10

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ConfigError("features.queue_persistence.window must be >= 2")
        _require_range(
            "features.queue_persistence.tolerance", self.tolerance, 0.0, 1.0
        )
        if not 1 <= self.min_samples <= self.window:
            raise ConfigError(
                "features.queue_persistence.min_samples must be in 1..window"
            )


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    """Pressure momentum parameters: fast minus slow EMA of the imbalance."""

    fast_half_life_ms: float = 600.0
    slow_half_life_ms: float = 3_000.0
    variance_window: int = 120
    dispersion_floor: float = 0.02
    dispersion_k: float = 2.0
    max_dt_ms: float = 5_000.0
    warmup_updates: int = 6

    def __post_init__(self) -> None:
        _require_positive("features.momentum.fast_half_life_ms", self.fast_half_life_ms)
        _require_ordered(
            "features.momentum.fast_half_life_ms",
            self.fast_half_life_ms,
            "features.momentum.slow_half_life_ms",
            self.slow_half_life_ms,
        )
        if self.variance_window < 2:
            raise ConfigError("features.momentum.variance_window must be >= 2")
        _require_positive("features.momentum.dispersion_floor", self.dispersion_floor)
        _require_positive("features.momentum.dispersion_k", self.dispersion_k)
        _require_positive("features.momentum.max_dt_ms", self.max_dt_ms)
        if self.warmup_updates < 1:
            raise ConfigError("features.momentum.warmup_updates must be >= 1")


@dataclass(frozen=True, slots=True)
class AccelerationConfig:
    """Pressure acceleration parameters: time derivative of momentum."""

    variance_window: int = 120
    dispersion_floor: float = 0.05
    dispersion_k: float = 2.0
    min_dt_ms: float = 20.0
    smooth_half_life_ms: float = 500.0
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        if self.variance_window < 2:
            raise ConfigError("features.acceleration.variance_window must be >= 2")
        _require_positive("features.acceleration.dispersion_floor", self.dispersion_floor)
        _require_positive("features.acceleration.dispersion_k", self.dispersion_k)
        _require_positive("features.acceleration.min_dt_ms", self.min_dt_ms)
        _require_positive(
            "features.acceleration.smooth_half_life_ms", self.smooth_half_life_ms
        )
        _require_positive("features.acceleration.max_dt_ms", self.max_dt_ms)


@dataclass(frozen=True, slots=True)
class RefillConfig:
    """Refill-proxy parameters.

    ``volume_confirmation_ratio`` is the fraction of the observed quantity drop
    that must be explained by an increase in the day's traded volume before the
    drop is classified as *consumption*. Without this test a cancellation is
    indistinguishable from a trade, and the two carry opposite information.
    """

    consumption_fraction: float = 0.35
    recovery_fraction: float = 0.60
    window_ms: float = 1_500.0
    volume_confirmation_ratio: float = 0.50
    decay_half_life_ms: float = 800.0
    failure_weight: float = 0.60
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        _require_range(
            "features.refill.consumption_fraction", self.consumption_fraction, 0.01, 1.0
        )
        _require_range(
            "features.refill.recovery_fraction", self.recovery_fraction, 0.01, 2.0
        )
        _require_positive("features.refill.window_ms", self.window_ms)
        _require_range(
            "features.refill.volume_confirmation_ratio",
            self.volume_confirmation_ratio,
            0.0,
            2.0,
        )
        _require_positive("features.refill.decay_half_life_ms", self.decay_half_life_ms)
        _require_range("features.refill.failure_weight", self.failure_weight, 0.0, 2.0)
        _require_positive("features.refill.max_dt_ms", self.max_dt_ms)


@dataclass(frozen=True, slots=True)
class LtpConfirmationConfig:
    """Last-traded-price confirmation parameters.

    ``opposition_threshold`` is the magnitude at which an opposing LTP drift
    becomes a veto rather than merely a low-weight disagreement.
    """

    half_life_ms: float = 1_200.0
    scale_ticks: float = 1.5
    stale_ms: float = 3_000.0
    opposition_threshold: float = 0.35
    position_weight: float = 0.5
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        _require_positive("features.ltp_confirmation.half_life_ms", self.half_life_ms)
        _require_positive("features.ltp_confirmation.scale_ticks", self.scale_ticks)
        _require_positive("features.ltp_confirmation.stale_ms", self.stale_ms)
        _require_range(
            "features.ltp_confirmation.opposition_threshold",
            self.opposition_threshold,
            0.0,
            1.0,
        )
        _require_range(
            "features.ltp_confirmation.position_weight", self.position_weight, 0.0, 1.0
        )
        _require_positive("features.ltp_confirmation.max_dt_ms", self.max_dt_ms)


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    """Container for all feature parameter blocks."""

    microprice: MicropriceConfig = field(default_factory=MicropriceConfig)
    weighted_obi: WeightedObiConfig = field(default_factory=WeightedObiConfig)
    depth_slope: DepthSlopeConfig = field(default_factory=DepthSlopeConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    spread_compression: SpreadCompressionConfig = field(
        default_factory=SpreadCompressionConfig
    )
    queue_persistence: QueuePersistenceConfig = field(
        default_factory=QueuePersistenceConfig
    )
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    refill: RefillConfig = field(default_factory=RefillConfig)
    ltp_confirmation: LtpConfirmationConfig = field(
        default_factory=LtpConfirmationConfig
    )


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureSensitivity:
    """Exponents describing how one feature's confidence reacts to conditions.

    Confidence is computed as ``prod(factor_i ** exponent_i)``. An exponent of
    zero makes a feature indifferent to that factor; larger exponents make it
    more sensitive. Using exponents on a product of factors in ``[0, 1]``
    guarantees the result stays in ``[0, 1]`` without any clipping, and makes
    the model monotone in every factor.

    Example of the intended asymmetry: a level feature such as the microprice
    degrades sharply as the spread widens (``spread`` exponent high) whereas
    momentum cares far more about having a stable, gap-free sample history
    (``stability`` and ``warmup`` exponents high).
    """

    spread: float = 1.0
    liquidity: float = 1.0
    volatility: float = 0.5
    book_quality: float = 1.0
    stability: float = 0.5
    warmup: float = 1.0

    def __post_init__(self) -> None:
        for name in ("spread", "liquidity", "volatility", "book_quality", "stability", "warmup"):
            value = getattr(self, name)
            _require_range(f"confidence.sensitivity.{name}", value, 0.0, 4.0)


#: Per-feature confidence sensitivities. Level-based features are dominated by
#: spread and liquidity; history-based features by stability and warmup.
DEFAULT_SENSITIVITIES: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {
        F_MICROPRICE: {"spread": 1.6, "liquidity": 1.0, "volatility": 0.4, "stability": 0.3},
        F_WEIGHTED_OBI: {"spread": 1.0, "liquidity": 1.4, "volatility": 0.5, "stability": 0.4},
        F_DEPTH_SLOPE: {"spread": 0.8, "liquidity": 1.6, "volatility": 0.5, "stability": 0.5},
        F_QUEUE_PERSISTENCE: {
            "spread": 0.6,
            "liquidity": 1.0,
            "volatility": 0.8,
            "stability": 1.2,
            "warmup": 1.4,
        },
        F_REFILL: {
            "spread": 0.8,
            "liquidity": 1.2,
            "volatility": 0.6,
            "stability": 0.8,
            "warmup": 1.2,
        },
        F_MOMENTUM: {
            "spread": 0.6,
            "liquidity": 0.8,
            "volatility": 0.9,
            "stability": 1.0,
            "warmup": 1.6,
        },
        F_ACCELERATION: {
            "spread": 0.5,
            "liquidity": 0.7,
            "volatility": 1.3,
            "stability": 1.2,
            "warmup": 1.8,
        },
        F_LTP_CONFIRMATION: {
            "spread": 0.9,
            "liquidity": 0.6,
            "volatility": 0.7,
            "stability": 0.6,
            "warmup": 1.0,
        },
    }
)

#: Confidence multiplier applied per regime. This — not the weight table — is
#: how the "in NOISE, trust everything less" rule is expressed: because the
#: composite is normalised by its own weight mass, scaling all weights down
#: would cancel out exactly and change nothing.
DEFAULT_REGIME_CONFIDENCE: Final[Mapping[str, float]] = MappingProxyType(
    {"TREND": 1.0, "PULLBACK": 0.85, "RANGE": 0.95, "NOISE": 0.45, "UNKNOWN": 0.5}
)


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Parameters of the shared confidence model."""

    spread_tight_ticks: float = 1.0
    spread_wide_ticks: float = 6.0
    liquidity_window: int = 240
    liquidity_floor_ratio: float = 0.20
    volatility_window: int = 240
    volatility_tolerance: float = 2.0
    stability_window: int = 60
    min_feature_confidence: float = 0.02
    sensitivities: Mapping[str, FeatureSensitivity] = field(
        default_factory=lambda: {
            name: FeatureSensitivity(**values)
            for name, values in DEFAULT_SENSITIVITIES.items()
        }
    )
    regime_multipliers: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REGIME_CONFIDENCE)
    )

    def __post_init__(self) -> None:
        _require_ordered(
            "confidence.spread_tight_ticks",
            self.spread_tight_ticks,
            "confidence.spread_wide_ticks",
            self.spread_wide_ticks,
        )
        if self.liquidity_window < 2:
            raise ConfigError("confidence.liquidity_window must be >= 2")
        if self.volatility_window < 2:
            raise ConfigError("confidence.volatility_window must be >= 2")
        if self.stability_window < 2:
            raise ConfigError("confidence.stability_window must be >= 2")
        _require_range(
            "confidence.liquidity_floor_ratio", self.liquidity_floor_ratio, 0.0, 1.0
        )
        _require_positive("confidence.volatility_tolerance", self.volatility_tolerance)
        _require_range(
            "confidence.min_feature_confidence", self.min_feature_confidence, 0.0, 0.5
        )
        for regime, multiplier in self.regime_multipliers.items():
            _require_range(f"confidence.regime_multipliers.{regime}", multiplier, 0.0, 1.0)
        _freeze_mapping(self, "sensitivities", self.sensitivities)
        _freeze_mapping(self, "regime_multipliers", self.regime_multipliers)

    def sensitivity_for(self, feature: str) -> FeatureSensitivity:
        """Return the sensitivity block for ``feature``, or a neutral default."""
        found = self.sensitivities.get(feature)
        return found if found is not None else FeatureSensitivity()


# --------------------------------------------------------------------------- #
# Regime, composite, threshold
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Deterministic regime detector parameters.

    The detector is built on Kaufman's efficiency ratio (net displacement
    divided by path length) plus realised volatility, both expressed in ticks.
    ``min_dwell_snapshots`` and ``confirm_snapshots`` provide hysteresis: without
    them the label flips on almost every snapshot, and the adaptive weights
    flip with it.
    """

    window: int = 90
    fast_half_life_ms: float = 1_500.0
    slow_half_life_ms: float = 8_000.0
    er_trend: float = 0.45
    er_range: float = 0.25
    vol_high_ticks: float = 2.5
    vol_low_ticks: float = 0.8
    trend_min_ticks: float = 1.0
    pullback_min_er: float = 0.30
    min_dwell_snapshots: int = 8
    confirm_snapshots: int = 3
    min_samples: int = 30
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ConfigError("regime.window must be >= 4")
        _require_ordered(
            "regime.fast_half_life_ms",
            self.fast_half_life_ms,
            "regime.slow_half_life_ms",
            self.slow_half_life_ms,
        )
        _require_range("regime.er_trend", self.er_trend, 0.0, 1.0)
        _require_range("regime.er_range", self.er_range, 0.0, 1.0)
        _require_ordered("regime.er_range", self.er_range, "regime.er_trend", self.er_trend)
        _require_ordered(
            "regime.vol_low_ticks",
            self.vol_low_ticks,
            "regime.vol_high_ticks",
            self.vol_high_ticks,
        )
        _require_non_negative("regime.trend_min_ticks", self.trend_min_ticks)
        _require_range("regime.pullback_min_er", self.pullback_min_er, 0.0, 1.0)
        if self.min_dwell_snapshots < 1:
            raise ConfigError("regime.min_dwell_snapshots must be >= 1")
        if self.confirm_snapshots < 1:
            raise ConfigError("regime.confirm_snapshots must be >= 1")
        if not 2 <= self.min_samples <= self.window:
            raise ConfigError("regime.min_samples must be in 2..window")
        _require_positive("regime.max_dt_ms", self.max_dt_ms)


#: Adaptive directional weights per regime. Only relative magnitudes matter,
#: because the composite divides by its own weight mass.
DEFAULT_REGIME_WEIGHTS: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {
        "TREND": {
            F_MOMENTUM: 0.22,
            F_ACCELERATION: 0.12,
            F_WEIGHTED_OBI: 0.20,
            F_MICROPRICE: 0.08,
            F_REFILL: 0.12,
            F_DEPTH_SLOPE: 0.08,
            F_QUEUE_PERSISTENCE: 0.06,
            F_LTP_CONFIRMATION: 0.12,
        },
        "PULLBACK": {
            F_MOMENTUM: 0.14,
            F_ACCELERATION: 0.08,
            F_WEIGHTED_OBI: 0.20,
            F_MICROPRICE: 0.12,
            F_REFILL: 0.16,
            F_DEPTH_SLOPE: 0.08,
            F_QUEUE_PERSISTENCE: 0.10,
            F_LTP_CONFIRMATION: 0.12,
        },
        "RANGE": {
            F_MOMENTUM: 0.08,
            F_ACCELERATION: 0.04,
            F_WEIGHTED_OBI: 0.24,
            F_MICROPRICE: 0.20,
            F_REFILL: 0.10,
            F_DEPTH_SLOPE: 0.12,
            F_QUEUE_PERSISTENCE: 0.12,
            F_LTP_CONFIRMATION: 0.10,
        },
        "NOISE": {
            F_MOMENTUM: 0.12,
            F_ACCELERATION: 0.06,
            F_WEIGHTED_OBI: 0.20,
            F_MICROPRICE: 0.16,
            F_REFILL: 0.12,
            F_DEPTH_SLOPE: 0.10,
            F_QUEUE_PERSISTENCE: 0.12,
            F_LTP_CONFIRMATION: 0.12,
        },
        "UNKNOWN": {
            F_MOMENTUM: 0.12,
            F_ACCELERATION: 0.06,
            F_WEIGHTED_OBI: 0.22,
            F_MICROPRICE: 0.18,
            F_REFILL: 0.10,
            F_DEPTH_SLOPE: 0.10,
            F_QUEUE_PERSISTENCE: 0.10,
            F_LTP_CONFIRMATION: 0.12,
        },
    }
)


@dataclass(frozen=True, slots=True)
class CompositeConfig:
    """Composite scoring parameters.

    The composite is ``sum(value * weight * confidence) / sum(weight *
    confidence)``. The normalisation is what keeps the score inside
    ``[-1, +1]`` and comparable across regimes and across confidence levels; an
    unnormalised sum would shrink whenever confidence fell and would then be
    compared against a threshold that also moves with market conditions,
    counting the same effect twice.

    ``collinear_groups`` caps the joint weight of features that are close to
    algebraically dependent. The normalised microprice tilt equals the L1
    imbalance exactly (``microprice - mid == (spread / 2) * obi_L1``), so
    microprice and the L1/L2 weighted imbalance must not be allowed to vote
    twice for the same observation.

    ``min_confidence`` is the minimum weighted-average confidence required for a
    score to be considered usable. It is deliberately expressed as an average in
    ``[0, 1]`` rather than as a sum of ``weight * confidence``: a sum depends on
    the absolute scale of the weight table, which would break the very
    scale-invariance the normalised composite exists to provide, and would make
    the validity of a score depend on whether the operator wrote weights that
    total 1.0 or 100.
    """

    weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            regime: dict(values) for regime, values in DEFAULT_REGIME_WEIGHTS.items()
        }
    )
    smoothing_half_life_ms: float = 700.0
    min_features: int = 3
    min_confidence: float = 0.15
    collinear_groups: Mapping[str, float] = field(
        default_factory=lambda: {f"{F_MICROPRICE}|{F_WEIGHTED_OBI}": 0.30}
    )
    max_dt_ms: float = 5_000.0

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigError("composite.weights must not be empty")
        for regime, table in self.weights.items():
            if not table:
                raise ConfigError(f"composite.weights.{regime} must not be empty")
            for name, weight in table.items():
                _require_range(f"composite.weights.{regime}.{name}", weight, 0.0, 10.0)
        _require_positive("composite.smoothing_half_life_ms", self.smoothing_half_life_ms)
        if self.min_features < 1:
            raise ConfigError("composite.min_features must be >= 1")
        _require_range("composite.min_confidence", self.min_confidence, 0.0, 1.0)
        for group, cap in self.collinear_groups.items():
            if "|" not in group:
                raise ConfigError(
                    "composite.collinear_groups keys must be 'featureA|featureB' "
                    f"pipe-separated names, got {group!r}"
                )
            _require_range(f"composite.collinear_groups.{group}", cap, 0.0, 10.0)
        _require_positive("composite.max_dt_ms", self.max_dt_ms)
        _freeze_mapping(
            self,
            "weights",
            {regime: MappingProxyType(dict(table)) for regime, table in self.weights.items()},
        )
        _freeze_mapping(self, "collinear_groups", self.collinear_groups)

    def weights_for(self, regime: str) -> Mapping[str, float]:
        """Return the weight table for ``regime``, falling back to ``UNKNOWN``."""
        table = self.weights.get(regime)
        if table is not None:
            return table
        fallback = self.weights.get("UNKNOWN")
        if fallback is not None:
            return fallback
        return next(iter(self.weights.values()))


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Adaptive threshold parameters.

    The threshold rises with spread volatility, mid volatility, imbalance
    volatility and queue instability, and is hard-clamped into
    ``[min_entry, max_entry]``. The clamp is not cosmetic: in a very quiet book
    every dispersion term collapses towards zero, and an unclamped adaptive
    threshold would converge on zero and fire on numerical dust.
    """

    base: float = 0.30
    spread_vol_coeff: float = 0.45
    mid_vol_coeff: float = 0.40
    obi_vol_coeff: float = 0.35
    instability_coeff: float = 0.30
    min_entry: float = 0.18
    max_entry: float = 0.85
    watch_ratio: float = 0.60
    exit_ratio: float = 0.50
    reference_window: int = 240
    imbalance_vol_reference: float = 0.25

    def __post_init__(self) -> None:
        _require_range("threshold.base", self.base, 0.0, 1.0)
        for name in (
            "spread_vol_coeff",
            "mid_vol_coeff",
            "obi_vol_coeff",
            "instability_coeff",
        ):
            _require_range(f"threshold.{name}", getattr(self, name), 0.0, 5.0)
        _require_range("threshold.min_entry", self.min_entry, 0.01, 1.0)
        _require_range("threshold.max_entry", self.max_entry, 0.01, 1.0)
        _require_ordered(
            "threshold.min_entry", self.min_entry, "threshold.max_entry", self.max_entry
        )
        _require_range("threshold.watch_ratio", self.watch_ratio, 0.05, 1.0)
        _require_range("threshold.exit_ratio", self.exit_ratio, 0.0, 1.0)
        if self.reference_window < 2:
            raise ConfigError("threshold.reference_window must be >= 2")
        _require_positive(
            "threshold.imbalance_vol_reference", self.imbalance_vol_reference
        )


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StateMachineConfig:
    """Entry, exit and timing rules."""

    watch_timeout_ms: float = 2_500.0
    entry_confirmations: int = 2
    min_watch_confidence: float = 0.35
    min_entry_confidence: float = 0.55
    exit_confidence: float = 0.25
    min_hold_ms: float = 250.0
    max_hold_ms: float = 60_000.0
    cooldown_ms: float = 1_500.0
    stop_ticks: float = 6.0
    target_ticks: float = 12.0
    reversal_ratio: float = 0.60
    require_ltp_confirmation: bool = True
    require_zero_cross_rearm: bool = True
    exit_on_quality_loss: bool = True
    use_target: bool = True

    def __post_init__(self) -> None:
        _require_positive("state_machine.watch_timeout_ms", self.watch_timeout_ms)
        if self.entry_confirmations < 1:
            raise ConfigError("state_machine.entry_confirmations must be >= 1")
        _require_range(
            "state_machine.min_watch_confidence", self.min_watch_confidence, 0.0, 1.0
        )
        _require_range(
            "state_machine.min_entry_confidence", self.min_entry_confidence, 0.0, 1.0
        )
        _require_range("state_machine.exit_confidence", self.exit_confidence, 0.0, 1.0)
        if self.min_watch_confidence > self.min_entry_confidence:
            raise ConfigError(
                "state_machine.min_watch_confidence must not exceed min_entry_confidence"
            )
        _require_non_negative("state_machine.min_hold_ms", self.min_hold_ms)
        _require_ordered(
            "state_machine.min_hold_ms",
            self.min_hold_ms,
            "state_machine.max_hold_ms",
            self.max_hold_ms,
        )
        _require_non_negative("state_machine.cooldown_ms", self.cooldown_ms)
        _require_positive("state_machine.stop_ticks", self.stop_ticks)
        _require_positive("state_machine.target_ticks", self.target_ticks)
        _require_range("state_machine.reversal_ratio", self.reversal_ratio, 0.0, 3.0)


# --------------------------------------------------------------------------- #
# Execution and costs (kept out of the signal path)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Round-trip transaction cost model, in Indian equity intraday terms.

    Defaults reflect a typical discount-broker intraday equity schedule and are
    expressed as configurable rates rather than constants in code, because they
    differ per broker and change with regulation. The model is only consulted by
    :mod:`engine.execution`; it can be disabled entirely and the signal path is
    identical either way.
    """

    enabled: bool = True
    brokerage_flat_per_order: float = 20.0
    brokerage_rate: float = 0.0003
    brokerage_cap_per_order: float = 20.0
    exchange_txn_rate: float = 0.0000297
    stt_sell_rate: float = 0.00025
    stamp_duty_buy_rate: float = 0.00003
    sebi_rate: float = 0.000001
    gst_rate: float = 0.18

    def __post_init__(self) -> None:
        for name in (
            "brokerage_flat_per_order",
            "brokerage_rate",
            "brokerage_cap_per_order",
            "exchange_txn_rate",
            "stt_sell_rate",
            "stamp_duty_buy_rate",
            "sebi_rate",
            "gst_rate",
        ):
            _require_non_negative(f"execution.cost.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Execution model parameters.

    ``entry_aggression_bps`` is converted to ticks against the live reference
    price and then rounded onto the tick grid, because a basis-point offset is
    not a tradable price on an exchange with a fixed price increment. Supplying
    ``entry_aggression_ticks`` overrides the basis-point form.

    ``enforce_cost_gate`` is the optional link between the cost model and
    trading: when enabled, entries are blocked unless the configured target
    clears the round-trip cost by ``min_edge_multiple``. It defaults to off so
    that the signal engine's behaviour is not implicitly coupled to a cost
    assumption.
    """

    entry_aggression_bps: float = 3.0
    entry_aggression_ticks: float | None = None
    exit_slippage_bps: float = 1.0
    exit_slippage_ticks: float | None = None
    use_depth_walk: bool = False
    allow_partial_fill: bool = True
    enforce_cost_gate: bool = False
    min_edge_multiple: float = 1.5
    cost: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        _require_non_negative("execution.entry_aggression_bps", self.entry_aggression_bps)
        _require_non_negative("execution.exit_slippage_bps", self.exit_slippage_bps)
        if self.entry_aggression_ticks is not None:
            _require_non_negative(
                "execution.entry_aggression_ticks", self.entry_aggression_ticks
            )
        if self.exit_slippage_ticks is not None:
            _require_non_negative(
                "execution.exit_slippage_ticks", self.exit_slippage_ticks
            )
        _require_positive("execution.min_edge_multiple", self.min_edge_multiple)


# --------------------------------------------------------------------------- #
# Runtime and adapter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Process-level parameters.

    ``queue_size`` bounds the hand-off between the websocket thread and the
    engine thread. When the queue is full the *oldest* snapshot is dropped: in a
    live book the newest photograph is strictly more useful than a stale one,
    and dropping is preferable to applying backpressure to the socket.

    ``render_fps`` decouples presentation from compute. Rendering inside the
    per-snapshot path would dominate the latency budget by orders of magnitude.
    A value of ``0`` disables the render thread entirely, which is what a
    headless run or a benchmark wants.
    """

    queue_size: int = 4096
    render_fps: float = 6.0
    log_level: str = "INFO"
    log_file: str | None = "logs/snapshot_quant_v4.log"
    log_console: bool = False
    latency_window: int = 8192
    stats_interval_ms: float = 15_000.0
    max_snapshots: int = 0
    renderer: str = "auto"

    def __post_init__(self) -> None:
        if self.queue_size < 8:
            raise ConfigError("runtime.queue_size must be >= 8")
        _require_range("runtime.render_fps", self.render_fps, 0.0, 60.0)
        if self.latency_window < 16:
            raise ConfigError("runtime.latency_window must be >= 16")
        _require_positive("runtime.stats_interval_ms", self.stats_interval_ms)
        _require_non_negative("runtime.max_snapshots", self.max_snapshots)
        if self.renderer not in ("auto", "rich", "plain", "none"):
            raise ConfigError(
                "runtime.renderer must be one of auto, rich, plain, none; "
                f"got {self.renderer!r}"
            )


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Websocket adapter parameters.

    Reconnection is a bounded exponential backoff with jitter, driven by a
    loop rather than by recursion: a recursive reconnect grows the stack for the
    lifetime of the process and eventually fails in exactly the situation where
    reliability matters most.
    """

    subscription_mode: int = int(SubscriptionMode.SNAP_QUOTE)
    correlation_id: str = "sq4"
    reconnect_initial_ms: float = 1_000.0
    reconnect_max_ms: float = 30_000.0
    reconnect_multiplier: float = 2.0
    reconnect_jitter_ratio: float = 0.25
    max_reconnect_attempts: int = 0
    heartbeat_timeout_ms: float = 30_000.0
    login_retry_attempts: int = 3
    totp_min_remaining_s: float = 2.0

    def __post_init__(self) -> None:
        if self.subscription_mode != int(SubscriptionMode.SNAP_QUOTE):
            raise ConfigError(
                "adapter.subscription_mode must be 3 (SnapQuote): this engine is "
                "built for snapshot mode only"
            )
        if not self.correlation_id.strip():
            raise ConfigError("adapter.correlation_id must be non-empty")
        _require_positive("adapter.reconnect_initial_ms", self.reconnect_initial_ms)
        _require_ordered(
            "adapter.reconnect_initial_ms",
            self.reconnect_initial_ms,
            "adapter.reconnect_max_ms",
            self.reconnect_max_ms,
        )
        if self.reconnect_multiplier < 1.0:
            raise ConfigError("adapter.reconnect_multiplier must be >= 1.0")
        _require_range(
            "adapter.reconnect_jitter_ratio", self.reconnect_jitter_ratio, 0.0, 1.0
        )
        _require_non_negative(
            "adapter.max_reconnect_attempts", self.max_reconnect_attempts
        )
        _require_positive("adapter.heartbeat_timeout_ms", self.heartbeat_timeout_ms)
        if self.login_retry_attempts < 1:
            raise ConfigError("adapter.login_retry_attempts must be >= 1")
        _require_non_negative("adapter.totp_min_remaining_s", self.totp_min_remaining_s)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Root configuration object."""

    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    symbols: tuple[SymbolConfig, ...] = ()
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for symbol in self.symbols:
            if symbol.token in seen:
                raise ConfigError(f"duplicate token in symbols: {symbol.token}")
            seen.add(symbol.token)

    def symbol_map(self) -> Mapping[str, SymbolConfig]:
        """Return a read-only token to :class:`SymbolConfig` mapping."""
        return MappingProxyType({symbol.token: symbol for symbol in self.symbols})


# --------------------------------------------------------------------------- #
# Strict deserialisation
# --------------------------------------------------------------------------- #

_ENV_PREFIX: Final[str] = "SQ4_"
_ENV_CREDENTIAL_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "API_KEY": "api_key",
        "CLIENT_CODE": "client_code",
        "PIN": "pin",
        "TOTP_SECRET": "totp_secret",
    }
)


def _is_dataclass_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convert a JSON value into ``annotation``, raising :class:`ConfigError`.

    Supported shapes are exactly those used by :class:`AppConfig`: primitives,
    ``X | None``, nested dataclasses, ``tuple[Dataclass, ...]``,
    ``Mapping[str, V]`` and nested mappings thereof. Anything else is a
    programming error in this module rather than a user error, and is reported
    as such.
    """
    origin = get_origin(annotation)

    if origin in (typing.Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if value is None:
            return None
        if len(args) != 1:
            raise ConfigError(f"{path}: unsupported union type {annotation!r}")
        return _coerce(value, args[0], path)

    if _is_dataclass_type(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
        return _build_dataclass(annotation, value, path)

    if origin is tuple:
        tuple_args = get_args(annotation)
        if len(tuple_args) != 2 or tuple_args[1] is not Ellipsis:
            raise ConfigError(f"{path}: unsupported tuple type {annotation!r}")
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(
            _coerce(item, tuple_args[0], f"{path}[{index}]")
            for index, item in enumerate(value)
        )

    # ``get_origin`` normalises ``typing.Mapping[...]`` to
    # ``collections.abc.Mapping``, so both spellings must be accepted here.
    if origin in (dict, abc.Mapping) or annotation in (dict, Mapping):
        mapping_args = get_args(annotation)
        value_type = mapping_args[1] if len(mapping_args) == 2 else Any
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
        return {
            str(key): _coerce(item, value_type, f"{path}.{key}")
            for key, item in value.items()
        }

    if annotation is Any:
        return value

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    raise ConfigError(f"{path}: unsupported configuration type {annotation!r}")


def _build_dataclass(cls: type, data: Mapping[str, Any], path: str) -> Any:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown key(s) {', '.join(unknown)}; "
            f"valid keys are {', '.join(sorted(field_names))}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        child_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(value, hints[name], child_path)
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path or cls.__name__}: {exc}") from exc


def _apply_environment(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Overlay credential values from the environment onto the parsed mapping.

    Environment variables win over the file so that a deployment can keep
    secrets out of ``config.json`` entirely.
    """
    credentials = data.setdefault("credentials", {})
    if not isinstance(credentials, dict):
        raise ConfigError("credentials: expected an object")
    for suffix, field_name in _ENV_CREDENTIAL_KEYS.items():
        env_value = environ.get(_ENV_PREFIX + suffix)
        if env_value:
            credentials[field_name] = env_value


def load_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load and validate ``config.json``.

    Parameters
    ----------
    path:
        Path to the JSON configuration file.
    environ:
        Environment mapping used for credential overlay. Defaults to
        ``os.environ``; injectable so the behaviour is testable.

    Raises
    ------
    ConfigError
        If the file is missing, is not an object, contains an unknown key, or
        holds an out-of-range value.
    """
    resolved = Path(path).expanduser()
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {resolved}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {resolved}: {exc}") from exc

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{resolved} is not valid JSON (line {exc.lineno}, column {exc.colno}): "
            f"{exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{resolved} must contain a JSON object at the top level")

    _apply_environment(parsed, os.environ if environ is None else environ)
    config: AppConfig = _build_dataclass(AppConfig, parsed, "")
    return config


def config_from_mapping(data: Mapping[str, Any]) -> AppConfig:
    """Build an :class:`AppConfig` from an in-memory mapping.

    Used by tests and by research notebooks that construct configurations
    programmatically rather than from disk.
    """
    config: AppConfig = _build_dataclass(AppConfig, data, "")
    return config
