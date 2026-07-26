"""Configuration loading and validation tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from snapshot_quant_v4.config import (
    AppConfig,
    CompositeConfig,
    ConfigError,
    CredentialsConfig,
    RegimeConfig,
    StateMachineConfig,
    SymbolConfig,
    ThresholdConfig,
    config_from_mapping,
    load_config,
)


def write(tmp_path: Path, payload: dict) -> Path:
    """Write a configuration file and return its path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoading:
    def test_minimal_configuration_uses_defaults(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"symbols": [{"token": "3045", "symbol": "SBIN"}]})
        config = load_config(path, environ={})
        assert isinstance(config, AppConfig)
        assert config.symbols[0].token == "3045"
        assert config.symbols[0].tick_size == pytest.approx(0.05)
        assert config.runtime.render_fps > 0.0
        assert config.adapter.subscription_mode == 3

    def test_nested_sections_are_merged_over_defaults(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {
                "symbols": [{"token": "3045"}],
                "features": {"weighted_obi": {"decay_ticks": 4.0}},
            },
        )
        config = load_config(path, environ={})
        assert config.features.weighted_obi.decay_ticks == pytest.approx(4.0)
        # Untouched siblings keep their defaults.
        assert config.features.weighted_obi.primary_levels == 2
        assert config.features.momentum.fast_half_life_ms > 0.0

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path, {"symbols": [{"token": "3045"}], "threshhold": {"base": 0.5}}
        )
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        assert "unknown key" in str(info.value)

    def test_unknown_nested_key_is_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {"symbols": [{"token": "3045"}], "threshold": {"bass": 0.5}},
        )
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        assert "threshold" in str(info.value)

    def test_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError) as info:
            load_config(tmp_path / "absent.json", environ={})
        assert "not found" in str(info.value)

    def test_invalid_json_is_reported_with_position(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        assert "not valid JSON" in str(info.value)

    def test_top_level_must_be_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path, environ={})

    def test_type_mismatch_is_reported(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {"symbols": [{"token": "3045"}], "runtime": {"queue_size": "many"}},
        )
        with pytest.raises(ConfigError) as info:
            load_config(path, environ={})
        assert "expected an integer" in str(info.value)

    def test_boolean_is_not_accepted_where_a_number_is_required(
        self, tmp_path: Path
    ) -> None:
        path = write(
            tmp_path, {"symbols": [{"token": "3045"}], "threshold": {"base": True}}
        )
        with pytest.raises(ConfigError):
            load_config(path, environ={})

    def test_optional_field_accepts_null(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {"symbols": [{"token": "3045"}], "runtime": {"log_file": None}},
        )
        config = load_config(path, environ={})
        assert config.runtime.log_file is None

    def test_nested_weight_tables_load(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {
                "symbols": [{"token": "3045"}],
                "composite": {"weights": {"TREND": {"momentum": 0.5, "refill": 0.5}}},
            },
        )
        config = load_config(path, environ={})
        assert config.composite.weights["TREND"]["momentum"] == pytest.approx(0.5)


class TestEnvironmentOverlay:
    def test_environment_supplies_credentials(self, tmp_path: Path) -> None:
        path = write(tmp_path, {"symbols": [{"token": "3045"}]})
        config = load_config(
            path,
            environ={
                "SQ4_API_KEY": "key-from-env",
                "SQ4_CLIENT_CODE": "AB1234",
                "SQ4_PIN": "1234",
                "SQ4_TOTP_SECRET": "GEZDGNBVGY3TQOJQ",
            },
        )
        assert config.credentials.api_key == "key-from-env"
        assert config.credentials.client_code == "AB1234"

    def test_environment_wins_over_the_file(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            {
                "symbols": [{"token": "3045"}],
                "credentials": {"api_key": "from-file"},
            },
        )
        config = load_config(path, environ={"SQ4_API_KEY": "from-env"})
        assert config.credentials.api_key == "from-env"


class TestImmutability:
    def test_mapping_fields_are_read_only(self) -> None:
        config = config_from_mapping({"symbols": [{"token": "3045"}]})
        with pytest.raises(TypeError):
            config.composite.weights["TREND"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            config.confidence.regime_multipliers["TREND"] = 0.0  # type: ignore[index]

    def test_sections_are_frozen(self) -> None:
        config = config_from_mapping({"symbols": [{"token": "3045"}]})
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.threshold.base = 0.9  # type: ignore[misc]


class TestRangeValidation:
    def test_symbol_requires_a_token(self) -> None:
        with pytest.raises(ConfigError):
            SymbolConfig(token="   ")

    def test_symbol_defaults_its_display_name_to_the_token(self) -> None:
        assert SymbolConfig(token="3045").symbol == "3045"

    def test_symbol_rejects_unknown_exchange(self) -> None:
        with pytest.raises(ConfigError):
            SymbolConfig(token="3045", exchange_type=99)

    def test_symbol_rejects_non_positive_tick(self) -> None:
        with pytest.raises(ConfigError):
            SymbolConfig(token="3045", tick_size=0.0)

    def test_duplicate_tokens_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            config_from_mapping(
                {"symbols": [{"token": "3045"}, {"token": "3045", "symbol": "dup"}]}
            )

    def test_threshold_bounds_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError):
            ThresholdConfig(min_entry=0.8, max_entry=0.2)

    def test_regime_efficiency_bands_must_be_ordered(self) -> None:
        with pytest.raises(ConfigError):
            RegimeConfig(er_range=0.8, er_trend=0.2)

    def test_state_machine_confidence_ordering(self) -> None:
        with pytest.raises(ConfigError):
            StateMachineConfig(min_watch_confidence=0.9, min_entry_confidence=0.4)

    def test_state_machine_hold_ordering(self) -> None:
        with pytest.raises(ConfigError):
            StateMachineConfig(min_hold_ms=10_000.0, max_hold_ms=1_000.0)

    def test_composite_group_key_must_be_pipe_separated(self) -> None:
        with pytest.raises(ConfigError):
            CompositeConfig(collinear_groups={"microprice": 0.3})

    def test_subscription_mode_must_be_snapquote(self) -> None:
        with pytest.raises(ConfigError):
            config_from_mapping(
                {"symbols": [{"token": "3045"}], "adapter": {"subscription_mode": 2}}
            )

    def test_renderer_choice_is_validated(self) -> None:
        with pytest.raises(ConfigError):
            config_from_mapping(
                {"symbols": [{"token": "3045"}], "runtime": {"renderer": "fancy"}}
            )


class TestCredentials:
    def test_secret_values_exclude_the_client_code(self) -> None:
        credentials = CredentialsConfig(
            api_key="key", client_code="AB1234", pin="9999", totp_secret="SEED"
        )
        secrets = credentials.secret_values()
        assert "key" in secrets
        assert "9999" in secrets
        assert "SEED" in secrets
        assert "AB1234" not in secrets

    def test_live_validation_lists_missing_fields(self) -> None:
        with pytest.raises(ConfigError) as info:
            CredentialsConfig(api_key="key").validate_for_live()
        message = str(info.value)
        assert "client_code" in message
        assert "pin" in message
        assert "totp_secret" in message

    def test_live_validation_passes_when_complete(self) -> None:
        CredentialsConfig(
            api_key="k", client_code="c", pin="p", totp_secret="s"
        ).validate_for_live()

    def test_replay_does_not_require_credentials(self) -> None:
        config = config_from_mapping({"symbols": [{"token": "3045"}]})
        assert config.credentials.api_key == ""


class TestSymbolMap:
    def test_symbol_map_is_read_only(self) -> None:
        config = config_from_mapping(
            {"symbols": [{"token": "3045"}, {"token": "1594"}]}
        )
        mapping = config.symbol_map()
        assert set(mapping) == {"3045", "1594"}
        with pytest.raises(TypeError):
            mapping["9999"] = SymbolConfig(token="9999")  # type: ignore[index]


class TestExampleConfig:
    def test_shipped_example_is_valid(self) -> None:
        example = Path(__file__).resolve().parents[1] / "config.example.json"
        if not example.is_file():
            pytest.skip("config.example.json is not present")
        config = load_config(example, environ={})
        assert config.symbols
        assert config.adapter.subscription_mode == 3
