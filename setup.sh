#!/usr/bin/env bash
# One-command Ubuntu/Debian VPS installer for Snapshot Quant Engine V4.
#
# Usage:
#   sudo bash setup.sh
#   sudo bash setup.sh --credentials-file /root/sq4.env --config /root/sq4.json
#   sudo bash setup.sh --reconfigure
#   sudo bash setup.sh --no-start
#
# This installs the live signal engine. The project models fills but does not
# place broker orders, reconcile broker positions, or guarantee square-off.

set -Eeuo pipefail
IFS=$'\n\t'
umask 0077

readonly SERVICE_NAME="snapshot-quant-v4"
readonly SERVICE_USER="snapshot-quant"
readonly SERVICE_GROUP="snapshot-quant"
readonly INSTALL_ROOT="/opt/${SERVICE_NAME}"
readonly RELEASES_DIR="${INSTALL_ROOT}/releases"
readonly CURRENT_LINK="${INSTALL_ROOT}/current"
readonly CONFIG_DIR="/etc/${SERVICE_NAME}"
readonly CONFIG_FILE="${CONFIG_DIR}/config.json"
readonly ENV_FILE="${CONFIG_DIR}/credentials.env"
readonly STATE_DIR="/var/lib/${SERVICE_NAME}"
readonly UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
readonly START_TIMER="/etc/systemd/system/${SERVICE_NAME}-start.timer"
readonly STOP_SERVICE="/etc/systemd/system/${SERVICE_NAME}-stop.service"
readonly STOP_TIMER="/etc/systemd/system/${SERVICE_NAME}-stop.timer"
readonly SESSION_CHECK="/usr/local/libexec/${SERVICE_NAME}-session-check"
readonly SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SMARTAPI_VERSION="1.5.0"
readonly WEBSOCKET_VERSION="1.9.0"
readonly REQUESTS_VERSION="2.32.5"
readonly CERTIFI_VERSION="2025.8.3"
readonly CHARSET_NORMALIZER_VERSION="3.4.3"
readonly IDNA_VERSION="3.10"
readonly URLLIB3_VERSION="2.5.0"
readonly PYTHON_DATEUTIL_VERSION="2.9.0.post0"
readonly SIX_VERSION="1.17.0"
readonly PYOTP_VERSION="2.9.0"
readonly LOGZERO_VERSION="1.7.0"
readonly PIP_VERSION="24.0"
readonly SETUPTOOLS_VERSION="75.8.0"
readonly WHEEL_VERSION="0.45.1"

RECONFIGURE=false
START_NOW=true
CHECK_ONLY=false
CREDENTIALS_INPUT=""
CONFIG_INPUT=""
WORK_DIR=""
CANDIDATE_DIR=""
CANDIDATE_CREATED=false
ACTIVATION_STARTED=false
ACTIVATION_COMPLETE=false
SMOKE_UNIT=""
SMOKE_CONFIG=""
PRIOR_CURRENT_TARGET=""
PRIOR_CURRENT_PRESENT=false
PRIOR_SERVICE_ACTIVE=false
PRIOR_SERVICE_ENABLED_STATE="not-found"
PRIOR_START_TIMER_ENABLED_STATE="not-found"
PRIOR_STOP_TIMER_ENABLED_STATE="not-found"
PRIOR_START_TIMER_ACTIVE=false
PRIOR_STOP_TIMER_ACTIVE=false

usage() {
    cat <<'EOF'
Usage: sudo bash setup.sh [OPTIONS]
       bash setup.sh --check

  --reconfigure           replace credentials and subscribed symbols while
                          preserving existing strategy tuning
  --credentials-file PATH noninteractive credentials input (four SQ4_* lines)
  --config PATH           noninteractive JSON input; on reconfigure only its
                          symbols replace the installed symbols
  --no-start              install and enable timers for the next boot, but keep
                          the service and timers inactive now
  --check                 run side-effect-free source/environment validation
  -h, --help              show this help

Supported platforms: Ubuntu 24.04+ and Debian 12+ with systemd.
Without input files, first setup/reconfiguration prompts in an SSH terminal.
The credentials file must contain exactly SQ4_API_KEY, SQ4_CLIENT_CODE, SQ4_PIN,
and SQ4_TOTP_SECRET as KEY=value or KEY="value" lines. It is parsed as data and
is never sourced as shell code.
EOF
}

log() {
    printf '[setup] %s\n' "$*"
}

fail() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '[setup] ERROR at line %s (exit %s)\n' "${BASH_LINENO[0]}" "${exit_code}" >&2
}
trap on_error ERR

unit_exists() {
    systemctl cat "$1" >/dev/null 2>&1
}

restore_backup() {
    local target=$1
    local name=$2
    local backup="${WORK_DIR}/backup/${name}"
    rm -f -- "${target}" || return 1
    if [[ -e "${backup}" || -L "${backup}" ]]; then
        cp -a -- "${backup}" "${target}" || return 1
        if [[ -L "${backup}" ]]; then
            [[ -L "${target}" && "$(readlink -- "${target}")" == "$(readlink -- "${backup}")" ]]
        else
            cmp -s -- "${backup}" "${target}"
        fi
    else
        [[ ! -e "${target}" && ! -L "${target}" ]]
    fi
}

restore_enablement() {
    local unit=$1
    local expected=$2
    local install_target="timers.target"
    if [[ "${unit}" == "${SERVICE_NAME}.service" ]]; then
        install_target="multi-user.target"
    fi
    # Remove links created by the attempted activation even when the restored
    # unit is absent or masked; systemctl disable cannot reliably clean a
    # dangling link for a no-longer-present unit.
    rm -f -- \
        "/etc/systemd/system/${install_target}.wants/${unit}" \
        "/run/systemd/system/${install_target}.wants/${unit}" || return 1
    case "${expected}" in
        enabled)
            systemctl enable "${unit}" >/dev/null
            ;;
        enabled-runtime)
            systemctl disable "${unit}" >/dev/null 2>&1 || true
            systemctl enable --runtime "${unit}" >/dev/null
            ;;
        disabled|static|indirect|not-found)
            if [[ "${expected}" != not-found ]]; then
                systemctl disable "${unit}" >/dev/null 2>&1 || true
            fi
            ;;
        masked|masked-runtime)
            # Restoring the backed-up /dev/null symlink restores the mask.
            ;;
        *)
            printf '[setup] rollback: unsupported prior state %s for %s\n' "${expected}" "${unit}" >&2
            return 1
            ;;
    esac
    local actual
    actual=$(systemctl is-enabled "${unit}" 2>/dev/null || true)
    [[ "${actual:-not-found}" == "${expected}" ]]
}

rollback_activation() {
    set +e
    local rollback_failed=false
    local restore_spec target name
    local actual_service_active actual_start_timer_active actual_stop_timer_active
    printf '[setup] ERROR: activation failed; restoring the previous installation\n' >&2
    systemctl stop "${SERVICE_NAME}.service" \
        "${SERVICE_NAME}-start.timer" "${SERVICE_NAME}-stop.timer" >/dev/null 2>&1

    for restore_spec in \
        "${ENV_FILE}|env" \
        "${CONFIG_FILE}|config" \
        "${SESSION_CHECK}|session-check" \
        "${UNIT_FILE}|service" \
        "${START_TIMER}|start-timer" \
        "${STOP_SERVICE}|stop-service" \
        "${STOP_TIMER}|stop-timer"; do
        target=${restore_spec%%|*}
        name=${restore_spec#*|}
        if ! restore_backup "${target}" "${name}"; then
            printf '[setup] rollback: failed to restore %s\n' "${target}" >&2
            rollback_failed=true
        fi
    done

    if ! rm -f -- "${CURRENT_LINK}"; then rollback_failed=true; fi
    if "${PRIOR_CURRENT_PRESENT}"; then
        if ! ln -s -- "${PRIOR_CURRENT_TARGET}" "${CURRENT_LINK}"; then
            rollback_failed=true
        fi
    fi
    if "${PRIOR_CURRENT_PRESENT}"; then
        if [[ ! -L "${CURRENT_LINK}" || "$(readlink -- "${CURRENT_LINK}" 2>/dev/null)" != "${PRIOR_CURRENT_TARGET}" ]]; then
            rollback_failed=true
        fi
    elif [[ -e "${CURRENT_LINK}" || -L "${CURRENT_LINK}" ]]; then
        rollback_failed=true
    fi

    if ! systemctl daemon-reload; then rollback_failed=true; fi
    if ! restore_enablement "${SERVICE_NAME}.service" "${PRIOR_SERVICE_ENABLED_STATE}"; then rollback_failed=true; fi
    if ! restore_enablement "${SERVICE_NAME}-start.timer" "${PRIOR_START_TIMER_ENABLED_STATE}"; then rollback_failed=true; fi
    if ! restore_enablement "${SERVICE_NAME}-stop.timer" "${PRIOR_STOP_TIMER_ENABLED_STATE}"; then rollback_failed=true; fi

    if "${PRIOR_START_TIMER_ACTIVE}"; then
        if ! systemctl start "${SERVICE_NAME}-start.timer"; then rollback_failed=true; fi
    fi
    if "${PRIOR_STOP_TIMER_ACTIVE}"; then
        if ! systemctl start "${SERVICE_NAME}-stop.timer"; then rollback_failed=true; fi
    fi
    if "${PRIOR_SERVICE_ACTIVE}"; then
        if ! systemctl start "${SERVICE_NAME}.service"; then rollback_failed=true; fi
    elif systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        if ! systemctl stop "${SERVICE_NAME}.service"; then rollback_failed=true; fi
    fi

    actual_service_active=false
    actual_start_timer_active=false
    actual_stop_timer_active=false
    if systemctl is-active --quiet "${SERVICE_NAME}.service"; then actual_service_active=true; fi
    if systemctl is-active --quiet "${SERVICE_NAME}-start.timer"; then actual_start_timer_active=true; fi
    if systemctl is-active --quiet "${SERVICE_NAME}-stop.timer"; then actual_stop_timer_active=true; fi
    if [[ "${actual_service_active}" != "${PRIOR_SERVICE_ACTIVE}" || \
          "${actual_start_timer_active}" != "${PRIOR_START_TIMER_ACTIVE}" || \
          "${actual_stop_timer_active}" != "${PRIOR_STOP_TIMER_ACTIVE}" ]]; then
        rollback_failed=true
    fi

    if "${rollback_failed}"; then
        printf '[setup] CRITICAL: rollback was incomplete; inspect systemd and %s immediately\n' "${CONFIG_DIR}" >&2
    else
        printf '[setup] previous installation and unit states verified as restored\n' >&2
    fi
}

cleanup() {
    local exit_code=$?
    set +e
    if [[ -n "${SMOKE_UNIT}" ]]; then
        systemctl stop "${SMOKE_UNIT}" >/dev/null 2>&1
        rm -f -- "/run/systemd/system/${SMOKE_UNIT}"
        systemctl daemon-reload >/dev/null 2>&1
        systemctl reset-failed "${SMOKE_UNIT}" >/dev/null 2>&1
    fi
    [[ -z "${SMOKE_CONFIG}" ]] || rm -f -- "${SMOKE_CONFIG}"

    if ((exit_code != 0)) && "${ACTIVATION_STARTED}" && ! "${ACTIVATION_COMPLETE}"; then
        rollback_activation
    fi
    if ((exit_code != 0)) && "${CANDIDATE_CREATED}" && [[ -n "${CANDIDATE_DIR}" ]]; then
        if [[ ! -L "${CURRENT_LINK}" || "$(readlink -- "${CURRENT_LINK}" 2>/dev/null)" != "${CANDIDATE_DIR}" ]]; then
            rm -rf -- "${CANDIDATE_DIR}"
        fi
    fi
    [[ -z "${WORK_DIR}" ]] || rm -rf -- "${WORK_DIR}"
    exit "${exit_code}"
}
trap cleanup EXIT

require_option_value() {
    local option=$1
    local value=${2-}
    [[ -n "${value}" && "${value}" != --* ]] || fail "${option} requires a path"
}

while (($#)); do
    case "$1" in
        --reconfigure)
            RECONFIGURE=true
            shift
            ;;
        --no-start)
            START_NOW=false
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --credentials-file)
            require_option_value "$1" "${2-}"
            CREDENTIALS_INPUT=$2
            shift 2
            ;;
        --config)
            require_option_value "$1" "${2-}"
            CONFIG_INPUT=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown option: $1"
            ;;
    esac
done

if "${CHECK_ONLY}" && { "${RECONFIGURE}" || ! "${START_NOW}" || [[ -n "${CREDENTIALS_INPUT}${CONFIG_INPUT}" ]]; }; then
    fail "--check cannot be combined with installation options"
fi

validate_source_tree() {
    local special=""
    [[ -f "${SOURCE_DIR}/pyproject.toml" ]] || fail "run this script from the repository"
    [[ -f "${SOURCE_DIR}/README.md" ]] || fail "README.md is missing"
    [[ -f "${SOURCE_DIR}/config.example.json" ]] || fail "config.example.json is missing"
    [[ -d "${SOURCE_DIR}/snapshot_quant_v4" ]] || fail "snapshot_quant_v4 package is missing"
    special=$(find \
        "${SOURCE_DIR}/pyproject.toml" \
        "${SOURCE_DIR}/README.md" \
        "${SOURCE_DIR}/config.example.json" \
        "${SOURCE_DIR}/snapshot_quant_v4" \
        \( -type l -o \( ! -type f ! -type d \) \) -print -quit)
    [[ -z "${special}" ]] || fail "runtime source contains a symlink or special file: ${special}"
    [[ -f "${SOURCE_DIR}/snapshot_quant_v4/__init__.py" ]] || fail "package __init__.py is missing"
}

validate_source_tree

if "${CHECK_ONLY}"; then
    command -v python3 >/dev/null || fail "python3 is not installed"
    python3 - <<'PY' || fail "Python 3.11+ is required"
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY
    python3 -m json.tool "${SOURCE_DIR}/config.example.json" >/dev/null
    python3 - "${SOURCE_DIR}/snapshot_quant_v4" <<'PY'
import ast
import sys
from pathlib import Path

for source in Path(sys.argv[1]).rglob("*.py"):
    ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
PY
    if command -v systemd-analyze >/dev/null; then
        systemd-analyze calendar 'Mon..Fri *-*-* 09:10:00 Asia/Kolkata' >/dev/null
        systemd-analyze calendar 'Mon..Fri *-*-* 15:35:00 Asia/Kolkata' >/dev/null
        log "CHECK PASS: allowlisted source, Python, JSON, AST, and IST calendars are valid"
    else
        log "CHECK SKIP: systemd-analyze is unavailable; timer calendars were not validated"
        log "CHECK PASS: allowlisted source, Python, JSON, and AST are valid"
    fi
    exit 0
fi

[[ ${EUID} -eq 0 ]] || fail "run as root: sudo bash setup.sh"
command -v systemctl >/dev/null || fail "systemd is required"
command -v apt-get >/dev/null || fail "an apt-based Ubuntu/Debian system is required"
[[ -d /run/systemd/system ]] || fail "systemd is not running as PID 1"
[[ -r /etc/os-release ]] || fail "/etc/os-release is unavailable"

OS_ID=""
OS_VERSION=""
while IFS='=' read -r key value; do
    value=${value%$'\r'}
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
        value=${value:1:${#value}-2}
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
        value=${value:1:${#value}-2}
    fi
    case "${key}" in
        ID) OS_ID=${value} ;;
        VERSION_ID) OS_VERSION=${value} ;;
    esac
done </etc/os-release

case "${OS_ID}" in
    ubuntu)
        [[ "${OS_VERSION}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "cannot parse Ubuntu VERSION_ID=${OS_VERSION}"
        ((10#${OS_VERSION%%.*} >= 24)) || fail "Ubuntu 24.04+ is required"
        ;;
    debian)
        [[ "${OS_VERSION}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "cannot parse Debian VERSION_ID=${OS_VERSION}"
        ((10#${OS_VERSION%%.*} >= 12)) || fail "Debian 12+ is required"
        ;;
    *)
        fail "unsupported platform ${OS_ID:-unknown}; use Ubuntu 24.04+ or Debian 12+"
        ;;
esac

command -v python3 >/dev/null || fail "Python 3.11+ must be present on the base image before setup"
python3 - <<'PY' || fail "Python 3.11+ must be present on the base image before setup"
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY

validate_regular_input() {
    local path=$1
    local label=$2
    local maximum=$3
    [[ -e "${path}" ]] || fail "${label} does not exist: ${path}"
    [[ ! -L "${path}" && -f "${path}" ]] || fail "${label} must be a regular, non-symlink file: ${path}"
    [[ -r "${path}" ]] || fail "${label} is not readable: ${path}"
    local size
    size=$(stat -c '%s' -- "${path}")
    ((size <= maximum)) || fail "${label} is unexpectedly large: ${path}"
}

if [[ -e "${CONFIG_FILE}" || -L "${CONFIG_FILE}" ]]; then
    validate_regular_input "${CONFIG_FILE}" "installed config" 1048576
fi
if [[ -e "${ENV_FILE}" || -L "${ENV_FILE}" ]]; then
    validate_regular_input "${ENV_FILE}" "installed credentials" 65536
fi
if [[ -n "${CREDENTIALS_INPUT}" ]]; then
    validate_regular_input "${CREDENTIALS_INPUT}" "credentials input" 65536
fi
if [[ -n "${CONFIG_INPUT}" ]]; then
    validate_regular_input "${CONFIG_INPUT}" "config input" 1048576
fi
if ! "${RECONFIGURE}" && [[ -e "${ENV_FILE}" && -n "${CREDENTIALS_INPUT}" ]]; then
    fail "credentials are already installed; add --reconfigure to replace them"
fi
if ! "${RECONFIGURE}" && [[ -e "${CONFIG_FILE}" && -n "${CONFIG_INPUT}" ]]; then
    fail "configuration is already installed; add --reconfigure to replace symbols"
fi

if getent passwd "${SERVICE_USER}" >/dev/null; then
    getent group "${SERVICE_GROUP}" >/dev/null || fail "existing ${SERVICE_USER} user has no dedicated ${SERVICE_GROUP} group"
    IFS=: read -r _ _ existing_uid existing_gid _ existing_home existing_shell < <(getent passwd "${SERVICE_USER}")
    service_gid=$(getent group "${SERVICE_GROUP}" | cut -d: -f3)
    group_members=$(getent group "${SERVICE_GROUP}" | cut -d: -f4)
    ((existing_uid < 1000)) || fail "existing ${SERVICE_USER} must be a system user"
    [[ "${existing_gid}" == "${service_gid}" ]] || fail "existing ${SERVICE_USER} must use ${SERVICE_GROUP} as its primary group"
    [[ "${existing_home}" == "${STATE_DIR}" ]] || fail "existing ${SERVICE_USER} has unexpected home ${existing_home}"
    [[ "${existing_shell}" == "/usr/sbin/nologin" ]] || fail "existing ${SERVICE_USER} has unexpected shell ${existing_shell}"
    [[ -z "${group_members}" || "${group_members}" == "${SERVICE_USER}" ]] || fail "${SERVICE_GROUP} has unexpected supplementary members"
    mapfile -t user_groups < <(id -Gn "${SERVICE_USER}" | tr ' ' '\n')
    ((${#user_groups[@]} == 1)) || fail "${SERVICE_USER} must not belong to supplementary groups"
elif getent group "${SERVICE_GROUP}" >/dev/null; then
    existing_group_gid=$(getent group "${SERVICE_GROUP}" | cut -d: -f3)
    group_members=$(getent group "${SERVICE_GROUP}" | cut -d: -f4)
    ((existing_group_gid < 1000)) || fail "existing ${SERVICE_GROUP} must be a system group"
    [[ -z "${group_members}" ]] || fail "existing ${SERVICE_GROUP} has unexpected members"
fi

prompt_value() {
    local variable_name=$1
    local prompt=$2
    local default_value=${3-}
    local secret=${4-false}
    local value=""

    [[ -t 0 ]] || fail "interactive input is required; use --credentials-file and --config for automation"
    if [[ "${secret}" == true ]]; then
        read -r -s -p "${prompt}: " value
        printf '\n'
    elif [[ -n "${default_value}" ]]; then
        read -r -p "${prompt} [${default_value}]: " value
        value=${value:-${default_value}}
    else
        read -r -p "${prompt}: " value
    fi
    [[ -n "${value}" ]] || fail "${prompt} cannot be empty"
    [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] || fail "invalid newline in ${prompt}"
    printf -v "${variable_name}" '%s' "${value}"
}

escape_environment_value() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '%s' "${value}"
}

WORK_DIR=$(mktemp -d "/run/${SERVICE_NAME}-setup.XXXXXX")
mkdir -p "${WORK_DIR}/backup" "${WORK_DIR}/verify-units" "${WORK_DIR}/final-units"
PREPARE_SCRIPT="${WORK_DIR}/prepare_config.py"
cat >"${PREPARE_SCRIPT}" <<'PY'
from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from pathlib import Path
from typing import NoReturn


def die(message: str) -> NoReturn:
    raise SystemExit(f"credentials/config validation failed: {message}")


def read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        die(f"{path} must contain a JSON object")
    return value


raw_path, env_path, base_path, output_path = map(Path, sys.argv[1:5])
symbol_path_text = sys.argv[5]
prompted = sys.argv[6] == "true"
allowed = {"SQ4_API_KEY", "SQ4_CLIENT_CODE", "SQ4_PIN", "SQ4_TOTP_SECRET"}
credentials: dict[str, str] = {}
for number, original in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
    line = original.strip()
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        die(f"credentials line {number} has no '='")
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in allowed:
        die(f"credentials line {number} has unsupported key {key!r}")
    if key in credentials:
        die(f"credentials key {key} is duplicated")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            die(f"credentials line {number} has invalid quoting: {exc.msg}")
        if not isinstance(decoded, str):
            die(f"credentials line {number} must contain a string")
        value = decoded
    elif any(char.isspace() for char in value) or any(char in value for char in "'\""):
        die(f"credentials line {number} must quote values containing spaces or quotes")
    if not value:
        die(f"credentials key {key} is empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        die(f"credentials key {key} contains a control character")
    credentials[key] = value

missing = sorted(allowed - credentials.keys())
if missing:
    die("missing credentials: " + ", ".join(missing))
secret = "".join(credentials["SQ4_TOTP_SECRET"].split()).upper().rstrip("=")
if len(secret) < 16 or re.fullmatch(r"[A-Z2-7]+", secret) is None:
    die("SQ4_TOTP_SECRET must be a base32 seed of at least 16 characters, not a 6-digit code")
try:
    decoded_secret = base64.b32decode(secret + "=" * ((8 - len(secret) % 8) % 8), casefold=False)
except binascii.Error as exc:
    die(f"SQ4_TOTP_SECRET is not valid base32: {exc}")
if len(decoded_secret) < 10:
    die("SQ4_TOTP_SECRET decodes to fewer than 80 bits")
credentials["SQ4_TOTP_SECRET"] = secret

env_path.write_text(
    "".join(f"{key}={json.dumps(credentials[key])}\n" for key in sorted(allowed)),
    encoding="utf-8",
)

data = read_object(base_path)
if symbol_path_text:
    symbol_path = Path(symbol_path_text)
    symbols_data = read_object(symbol_path).get("symbols")
    if not isinstance(symbols_data, list) or not symbols_data:
        die(f"{symbol_path} must contain a non-empty symbols list")
    data["symbols"] = symbols_data
elif prompted:
    if len(sys.argv) != 12:
        die("internal prompted-symbol argument error")
    try:
        exchange_type = int(sys.argv[9])
        tick_size = float(sys.argv[10])
        quantity = int(sys.argv[11])
    except ValueError as exc:
        die(f"invalid prompted symbol number: {exc}")
    data["symbols"] = [{
        "token": sys.argv[7].strip(),
        "symbol": sys.argv[8].strip().upper(),
        "exchange_type": exchange_type,
        "tick_size": tick_size,
        "quantity": quantity,
    }]

# Secrets always live in EnvironmentFile, never config.json.
data["credentials"] = {"api_key": "", "client_code": "", "pin": "", "totp_secret": ""}
runtime = data.setdefault("runtime", {})
if not isinstance(runtime, dict):
    die("runtime must be an object")
runtime.update({
    "render_fps": 0.0,
    "log_file": None,
    "log_console": True,
    "renderer": "none",
    "max_snapshots": 0,
})
output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
chmod 0700 "${PREPARE_SCRIPT}"

RAW_CREDENTIALS="${WORK_DIR}/credentials.input"
if "${RECONFIGURE}" || [[ ! -e "${ENV_FILE}" ]]; then
    if [[ -n "${CREDENTIALS_INPUT}" ]]; then
        cp -- "${CREDENTIALS_INPUT}" "${RAW_CREDENTIALS}"
    else
        prompt_value api_key "SmartAPI API key" "" true
        prompt_value client_code "SmartAPI client code"
        prompt_value pin "SmartAPI PIN/password" "" true
        prompt_value totp_secret "SmartAPI TOTP base32 secret" "" true
        cat >"${RAW_CREDENTIALS}" <<EOF
SQ4_API_KEY="$(escape_environment_value "${api_key}")"
SQ4_CLIENT_CODE="$(escape_environment_value "${client_code}")"
SQ4_PIN="$(escape_environment_value "${pin}")"
SQ4_TOTP_SECRET="$(escape_environment_value "${totp_secret}")"
EOF
        unset api_key client_code pin totp_secret
    fi
else
    cp -- "${ENV_FILE}" "${RAW_CREDENTIALS}"
fi

BASE_CONFIG="${WORK_DIR}/config.base.json"
SYMBOL_CONFIG=""
PROMPT_SYMBOL=false
if [[ -e "${CONFIG_FILE}" ]]; then
    cp -- "${CONFIG_FILE}" "${BASE_CONFIG}"
elif [[ -n "${CONFIG_INPUT}" ]]; then
    cp -- "${CONFIG_INPUT}" "${BASE_CONFIG}"
else
    cp -- "${SOURCE_DIR}/config.example.json" "${BASE_CONFIG}"
fi

if "${RECONFIGURE}" || [[ ! -e "${CONFIG_FILE}" ]]; then
    if [[ -n "${CONFIG_INPUT}" ]]; then
        SYMBOL_CONFIG="${WORK_DIR}/symbols.input.json"
        cp -- "${CONFIG_INPUT}" "${SYMBOL_CONFIG}"
    else
        PROMPT_SYMBOL=true
        prompt_value token "Angel One symbol token" "3045"
        prompt_value symbol "Display symbol" "SBIN"
        prompt_value exchange_type "Exchange type (NSE cash = 1)" "1"
        prompt_value tick_size "Tick size" "0.05"
        prompt_value quantity "Modelled intraday quantity" "1"
    fi
fi

CANONICAL_ENV="${WORK_DIR}/credentials.env"
CANDIDATE_CONFIG="${WORK_DIR}/config.json"
SYMBOL_ARGS=()
if "${PROMPT_SYMBOL}"; then
    SYMBOL_ARGS=("${token}" "${symbol}" "${exchange_type}" "${tick_size}" "${quantity}")
fi

log "validating credentials and merged configuration before host changes"
python3 "${PREPARE_SCRIPT}" \
    "${RAW_CREDENTIALS}" "${CANONICAL_ENV}" "${BASE_CONFIG}" \
    "${CANDIDATE_CONFIG}" "${SYMBOL_CONFIG}" "${PROMPT_SYMBOL}" \
    "${SYMBOL_ARGS[@]}"
PYTHONPATH="${SOURCE_DIR}" python3 - \
    "${CANDIDATE_CONFIG}" "${CANONICAL_ENV}" <<'PY'
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from snapshot_quant_v4.config import load_config


def parse_environment(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split("=", 1)
        value = json.loads(raw)
        if not isinstance(value, str):
            raise TypeError(f"{key} is not a string")
        parsed[key] = value
    return parsed


environ = parse_environment(Path(sys.argv[2]))
config = load_config(Path(sys.argv[1]), environ=environ)
config.credentials.validate_for_live()
secret = config.credentials.totp_secret
base64.b32decode(secret + "=" * ((8 - len(secret) % 8) % 8), casefold=False)
if not config.symbols:
    raise ValueError("live configuration has no symbols")
PY
chmod 0600 "${CANONICAL_ENV}" "${CANDIDATE_CONFIG}"

# All operator input and semantic validation has completed before package,
# account, directory, or service changes begin.
export DEBIAN_FRONTEND=noninteractive
log "installing required OS packages"
apt-get update -qq
apt-get install -y -qq ca-certificates python3 python3-venv python3-pip tzdata util-linux >/dev/null
python3 - <<'PY' || fail "Python 3.11+ is required; use Ubuntu 24.04+ or Debian 12+"
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY

if ! getent group "${SERVICE_GROUP}" >/dev/null; then
    log "creating restricted service group"
    groupadd --system "${SERVICE_GROUP}"
fi
if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
    log "creating restricted service account"
    useradd \
        --system \
        --gid "${SERVICE_GROUP}" \
        --home-dir "${STATE_DIR}" \
        --create-home \
        --shell /usr/sbin/nologin \
        "${SERVICE_USER}"
fi

ensure_directory() {
    local path=$1
    local mode=$2
    local owner=$3
    local group=$4
    [[ ! -L "${path}" ]] || fail "managed directory must not be a symlink: ${path}"
    if [[ -e "${path}" && ! -d "${path}" ]]; then
        fail "managed path is not a directory: ${path}"
    fi
    install -d -m "${mode}" -o "${owner}" -g "${group}" "${path}"
    chown "${owner}:${group}" "${path}"
    chmod "${mode}" "${path}"
}

ensure_directory "${INSTALL_ROOT}" 0755 root root
ensure_directory "${RELEASES_DIR}" 0755 root root
ensure_directory "${CONFIG_DIR}" 0750 root "${SERVICE_GROUP}"
ensure_directory "${STATE_DIR}" 0750 "${SERVICE_USER}" "${SERVICE_GROUP}"
ensure_directory /usr/local/libexec 0755 root root

# Preserved files are normalized even when no values are changed.
if [[ -e "${ENV_FILE}" ]]; then
    chown root:"${SERVICE_GROUP}" "${ENV_FILE}"
    chmod 0640 "${ENV_FILE}"
fi
if [[ -e "${CONFIG_FILE}" ]]; then
    chown root:"${SERVICE_GROUP}" "${CONFIG_FILE}"
    chmod 0640 "${CONFIG_FILE}"
fi

release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CANDIDATE_DIR="${RELEASES_DIR}/${release_id}"
[[ ! -e "${CANDIDATE_DIR}" ]] || fail "release path already exists: ${CANDIDATE_DIR}"
install -d -m 0755 -o root -g root "${CANDIDATE_DIR}"
CANDIDATE_CREATED=true

log "copying the explicit runtime allowlist into release ${release_id}"
install -m 0644 -o root -g root "${SOURCE_DIR}/pyproject.toml" "${CANDIDATE_DIR}/pyproject.toml"
install -m 0644 -o root -g root "${SOURCE_DIR}/README.md" "${CANDIDATE_DIR}/README.md"
install -m 0644 -o root -g root "${SOURCE_DIR}/config.example.json" "${CANDIDATE_DIR}/config.example.json"
python_files=0
while IFS= read -r -d '' source_file; do
    relative_file=${source_file#"${SOURCE_DIR}/"}
    destination_file="${CANDIDATE_DIR}/${relative_file}"
    install -d -m 0755 -o root -g root "$(dirname -- "${destination_file}")"
    install -m 0644 -o root -g root "${source_file}" "${destination_file}"
    ((python_files += 1))
done < <(find "${SOURCE_DIR}/snapshot_quant_v4" -type f -name '*.py' -print0)
((python_files > 0)) || fail "no Python runtime files were found"
chown -R root:root "${CANDIDATE_DIR}"
chmod -R go-w "${CANDIDATE_DIR}"

log "building a fresh pinned live environment"
python3 -m venv "${CANDIDATE_DIR}/.venv"
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
"${CANDIDATE_DIR}/.venv/bin/python" -m pip install --quiet \
    "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}"
"${CANDIDATE_DIR}/.venv/bin/python" -m pip install --quiet --no-deps \
    "smartapi-python==${SMARTAPI_VERSION}" \
    "websocket-client==${WEBSOCKET_VERSION}" \
    "requests==${REQUESTS_VERSION}" \
    "certifi==${CERTIFI_VERSION}" \
    "charset-normalizer==${CHARSET_NORMALIZER_VERSION}" \
    "idna==${IDNA_VERSION}" \
    "urllib3==${URLLIB3_VERSION}" \
    "python-dateutil==${PYTHON_DATEUTIL_VERSION}" \
    "six==${SIX_VERSION}" \
    "pyotp==${PYOTP_VERSION}" \
    "logzero==${LOGZERO_VERSION}"
"${CANDIDATE_DIR}/.venv/bin/python" -m pip check
"${CANDIDATE_DIR}/.venv/bin/python" -m pip install --quiet \
    --no-deps --no-build-isolation "${CANDIDATE_DIR}"
chown -R root:root "${CANDIDATE_DIR}"
# The installer keeps umask 0077 to protect credentials. venv/pip therefore
# create private directories unless runtime access is normalized explicitly.
# Candidate releases contain only allowlisted source and pinned dependencies;
# no credentials are stored below /opt.
chmod -R u=rwX,go=rX "${CANDIDATE_DIR}"
runuser -u "${SERVICE_USER}" -- test -x "${CANDIDATE_DIR}/.venv/bin/python" \
    || fail "service account cannot execute the candidate Python runtime"

CANONICAL_ENV="${WORK_DIR}/credentials.env"
CANDIDATE_CONFIG="${WORK_DIR}/config.json"
SYMBOL_ARGS=()
if "${PROMPT_SYMBOL}"; then
    SYMBOL_ARGS=("${token}" "${symbol}" "${exchange_type}" "${tick_size}" "${quantity}")
fi

"${CANDIDATE_DIR}/.venv/bin/python" - \
    "${RAW_CREDENTIALS}" "${CANONICAL_ENV}" "${BASE_CONFIG}" \
    "${CANDIDATE_CONFIG}" "${SYMBOL_CONFIG}" "${PROMPT_SYMBOL}" \
    "${SYMBOL_ARGS[@]}" <<'PY'
from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from pathlib import Path
from typing import NoReturn


def die(message: str) -> NoReturn:
    raise SystemExit(f"credentials/config validation failed: {message}")


def read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        die(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        die(f"{path} must contain a JSON object")
    return value


raw_path, env_path, base_path, output_path, symbol_path = map(Path, sys.argv[1:6])
prompted = sys.argv[6] == "true"
allowed = {"SQ4_API_KEY", "SQ4_CLIENT_CODE", "SQ4_PIN", "SQ4_TOTP_SECRET"}
credentials: dict[str, str] = {}
for number, original in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
    line = original.strip()
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        die(f"credentials line {number} has no '='")
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in allowed:
        die(f"credentials line {number} has unsupported key {key!r}")
    if key in credentials:
        die(f"credentials key {key} is duplicated")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            die(f"credentials line {number} has invalid quoting: {exc.msg}")
        if not isinstance(decoded, str):
            die(f"credentials line {number} must contain a string")
        value = decoded
    elif any(char.isspace() for char in value) or any(char in value for char in "'\""):
        die(f"credentials line {number} must quote values containing spaces or quotes")
    if not value:
        die(f"credentials key {key} is empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        die(f"credentials key {key} contains a control character")
    credentials[key] = value

missing = sorted(allowed - credentials.keys())
if missing:
    die("missing credentials: " + ", ".join(missing))
secret = "".join(credentials["SQ4_TOTP_SECRET"].split()).upper().rstrip("=")
if len(secret) < 16 or re.fullmatch(r"[A-Z2-7]+", secret) is None:
    die("SQ4_TOTP_SECRET must be a base32 seed of at least 16 characters, not a 6-digit code")
try:
    decoded_secret = base64.b32decode(secret + "=" * ((8 - len(secret) % 8) % 8), casefold=False)
except binascii.Error as exc:
    die(f"SQ4_TOTP_SECRET is not valid base32: {exc}")
if len(decoded_secret) < 10:
    die("SQ4_TOTP_SECRET decodes to fewer than 80 bits")
credentials["SQ4_TOTP_SECRET"] = secret

env_path.write_text(
    "".join(f"{key}={json.dumps(credentials[key])}\n" for key in sorted(allowed)),
    encoding="utf-8",
)

data = read_object(base_path)
if symbol_path.name:
    symbols_data = read_object(symbol_path).get("symbols")
    if not isinstance(symbols_data, list) or not symbols_data:
        die(f"{symbol_path} must contain a non-empty symbols list")
    data["symbols"] = symbols_data
elif prompted:
    if len(sys.argv) != 12:
        die("internal prompted-symbol argument error")
    try:
        exchange_type = int(sys.argv[9])
        tick_size = float(sys.argv[10])
        quantity = int(sys.argv[11])
    except ValueError as exc:
        die(f"invalid prompted symbol number: {exc}")
    data["symbols"] = [{
        "token": sys.argv[7].strip(),
        "symbol": sys.argv[8].strip().upper(),
        "exchange_type": exchange_type,
        "tick_size": tick_size,
        "quantity": quantity,
    }]

# Secrets always live in EnvironmentFile, never config.json.
data["credentials"] = {"api_key": "", "client_code": "", "pin": "", "totp_secret": ""}
runtime = data.setdefault("runtime", {})
if not isinstance(runtime, dict):
    die("runtime must be an object")
runtime.update({
    "render_fps": 0.0,
    "log_file": None,
    "log_console": True,
    "renderer": "none",
    "max_snapshots": 0,
})
output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
unset token symbol exchange_type tick_size quantity 2>/dev/null || true
chmod 0600 "${CANONICAL_ENV}" "${CANDIDATE_CONFIG}"

LIVE_PREFLIGHT_SCRIPT="${WORK_DIR}/live_preflight.py"
cat >"${LIVE_PREFLIGHT_SCRIPT}" <<'PY'
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from snapshot_quant_v4.adapter.angel_v2 import _import_smartapi
from snapshot_quant_v4.config import load_config


def parse_environment(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split("=", 1)
        value = json.loads(raw)
        if not isinstance(value, str):
            raise TypeError(f"{key} is not a string")
        parsed[key] = value
    return parsed


environ = parse_environment(Path(sys.argv[2]))
config = load_config(Path(sys.argv[1]), environ=environ)
config.credentials.validate_for_live()
secret = config.credentials.totp_secret
base64.b32decode(secret + "=" * ((8 - len(secret) % 8) % 8), casefold=False)
_import_smartapi()
if not config.symbols:
    raise ValueError("live configuration has no symbols")
PY
chown root:"${SERVICE_GROUP}" "${WORK_DIR}" "${CANONICAL_ENV}" \
    "${CANDIDATE_CONFIG}" "${LIVE_PREFLIGHT_SCRIPT}"
chmod 0750 "${WORK_DIR}"
chmod 0640 "${CANONICAL_ENV}" "${CANDIDATE_CONFIG}"
chmod 0550 "${LIVE_PREFLIGHT_SCRIPT}"

make_session_check() {
    local output=$1
    cat >"${output}" <<'EOF'
#!/bin/sh
# Weekday IST guard only. This is not an exchange-holiday calendar.
day=$(TZ=Asia/Kolkata date +%u) || exit 1
hhmm=$(TZ=Asia/Kolkata date +%H%M) || exit 1
[ "${day}" -le 5 ] && [ "${hhmm}" -ge 0910 ] && [ "${hhmm}" -lt 1535 ]
EOF
    chmod 0755 "${output}"
}

make_service_unit() {
    local output=$1
    local app_root=$2
    local config_path=$3
    local env_path=$4
    local session_path=$5
    cat >"${output}" <<EOF
[Unit]
Description=Snapshot Quant Engine V4 live signal service
Documentation=https://github.com/baldaurathore92-svg/Quantitative-Claude
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${STATE_DIR}
EnvironmentFile=${env_path}
Environment=PYTHONUNBUFFERED=1
ExecCondition=${session_path}
ExecStart=${app_root}/.venv/bin/snapshot-quant-v4 --config ${config_path} --mode live --renderer none --fps 0 --log-console
Restart=always
RestartSec=10
TimeoutStopSec=20
KillSignal=SIGTERM
UMask=0027
NoNewPrivileges=true
CapabilityBoundingSet=
AmbientCapabilities=
PrivateTmp=true
PrivateDevices=true
ProtectSystem=strict
ProtectHome=true
ProtectProc=invisible
ProcSubset=pid
ProtectHostname=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
RestrictNamespaces=true
RestrictSUIDSGID=true
RestrictRealtime=true
LockPersonality=true
MemoryDenyWriteExecute=true
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
DevicePolicy=closed
RemoveIPC=true
KeyringMode=private
StateDirectory=${SERVICE_NAME}
StateDirectoryMode=0750
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

make_start_timer() {
    local output=$1
    cat >"${output}" <<EOF
[Unit]
Description=Start Snapshot Quant Engine in the weekday IST window

[Timer]
OnCalendar=Mon..Fri *-*-* 09:10:00 Asia/Kolkata
Persistent=true
AccuracySec=30s
RandomizedDelaySec=0
Unit=${SERVICE_NAME}.service

[Install]
WantedBy=timers.target
EOF
}

make_stop_service() {
    local output=$1
    local systemctl_path=$2
    cat >"${output}" <<EOF
[Unit]
Description=Stop Snapshot Quant Engine after the weekday IST window

[Service]
Type=oneshot
User=root
Group=root
ExecStart=${systemctl_path} stop ${SERVICE_NAME}.service
UMask=0077
NoNewPrivileges=true
CapabilityBoundingSet=
AmbientCapabilities=
PrivateTmp=true
PrivateDevices=true
ProtectSystem=strict
ProtectHome=true
ProtectProc=invisible
ProcSubset=pid
ProtectHostname=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
RestrictNamespaces=true
RestrictSUIDSGID=true
RestrictRealtime=true
LockPersonality=true
MemoryDenyWriteExecute=true
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX
DevicePolicy=closed
RemoveIPC=true
KeyringMode=private
EOF
}

make_stop_timer() {
    local output=$1
    cat >"${output}" <<EOF
[Unit]
Description=Stop Snapshot Quant Engine after the weekday IST window

[Timer]
OnCalendar=Mon..Fri *-*-* 15:35:00 Asia/Kolkata
Persistent=true
AccuracySec=30s
RandomizedDelaySec=0
Unit=${SERVICE_NAME}-stop.service

[Install]
WantedBy=timers.target
EOF
}

VERIFY_DIR="${WORK_DIR}/verify-units"
FINAL_DIR="${WORK_DIR}/final-units"
# Some hardened VPS images mount /run with noexec. systemd-analyze checks
# ExecCondition executability, so stage the temporary helper under /opt beside
# the candidate release rather than in the noexec-capable work directory.
VERIFY_SESSION="${CANDIDATE_DIR}/.install-session-check"
make_session_check "${VERIFY_SESSION}"
make_session_check "${FINAL_DIR}/session-check"
make_service_unit "${VERIFY_DIR}/${SERVICE_NAME}.service" "${CANDIDATE_DIR}" \
    "${CANDIDATE_CONFIG}" "${CANONICAL_ENV}" "${VERIFY_SESSION}"
make_service_unit "${FINAL_DIR}/${SERVICE_NAME}.service" "${CURRENT_LINK}" \
    "${CONFIG_FILE}" "${ENV_FILE}" "${SESSION_CHECK}"
make_start_timer "${VERIFY_DIR}/${SERVICE_NAME}-start.timer"
make_start_timer "${FINAL_DIR}/${SERVICE_NAME}-start.timer"
SYSTEMCTL_PATH=$(command -v systemctl)
make_stop_service "${VERIFY_DIR}/${SERVICE_NAME}-stop.service" "${SYSTEMCTL_PATH}"
make_stop_service "${FINAL_DIR}/${SERVICE_NAME}-stop.service" "${SYSTEMCTL_PATH}"
make_stop_timer "${VERIFY_DIR}/${SERVICE_NAME}-stop.timer"
make_stop_timer "${FINAL_DIR}/${SERVICE_NAME}-stop.timer"
chmod 0644 "${VERIFY_DIR}"/* "${FINAL_DIR}"/*.service "${FINAL_DIR}"/*.timer

log "verifying candidate systemd units"
systemd-analyze verify \
    "${VERIFY_DIR}/${SERVICE_NAME}.service" \
    "${VERIFY_DIR}/${SERVICE_NAME}-start.timer" \
    "${VERIFY_DIR}/${SERVICE_NAME}-stop.service" \
    "${VERIFY_DIR}/${SERVICE_NAME}-stop.timer"
rm -f -- "${VERIFY_SESSION}"

log "running the no-network live preflight and 20-snapshot smoke test inside the systemd sandbox"
SMOKE_CONFIG="${CONFIG_DIR}/.candidate-${release_id}.json"
install -m 0640 -o root -g "${SERVICE_GROUP}" "${CANDIDATE_CONFIG}" "${SMOKE_CONFIG}"
SMOKE_UNIT="${SERVICE_NAME}-install-smoke-${release_id}.service"
cat >"/run/systemd/system/${SMOKE_UNIT}" <<EOF
[Unit]
Description=Snapshot Quant installer sandbox smoke test

[Service]
Type=oneshot
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${STATE_DIR}
EnvironmentFile=${CANONICAL_ENV}
ExecStartPre=${CANDIDATE_DIR}/.venv/bin/python ${LIVE_PREFLIGHT_SCRIPT} ${CANDIDATE_CONFIG} ${CANONICAL_ENV}
ExecStart=${CANDIDATE_DIR}/.venv/bin/snapshot-quant-v4 --config ${SMOKE_CONFIG} --mode synthetic --count 20 --max-snapshots 20 --renderer none --fps 0 --log-level ERROR
UMask=0027
NoNewPrivileges=true
CapabilityBoundingSet=
AmbientCapabilities=
PrivateTmp=true
PrivateDevices=true
ProtectSystem=strict
ProtectHome=true
ProtectProc=invisible
ProcSubset=pid
ProtectHostname=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectKernelLogs=true
ProtectControlGroups=true
ProtectClock=true
RestrictNamespaces=true
RestrictSUIDSGID=true
RestrictRealtime=true
LockPersonality=true
MemoryDenyWriteExecute=true
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX
DevicePolicy=closed
RemoveIPC=true
KeyringMode=private
StateDirectory=${SERVICE_NAME}
StateDirectoryMode=0750
StandardOutput=journal
StandardError=journal
EOF
chmod 0644 "/run/systemd/system/${SMOKE_UNIT}"
systemctl daemon-reload
systemctl start "${SMOKE_UNIT}"
systemctl reset-failed "${SMOKE_UNIT}" >/dev/null 2>&1 || true
rm -f -- "/run/systemd/system/${SMOKE_UNIT}" "${SMOKE_CONFIG}"
SMOKE_UNIT=""
SMOKE_CONFIG=""
systemctl daemon-reload

backup_target() {
    local target=$1
    local name=$2
    if [[ -e "${target}" || -L "${target}" ]]; then
        cp -a -- "${target}" "${WORK_DIR}/backup/${name}"
    fi
}

backup_target "${ENV_FILE}" env
backup_target "${CONFIG_FILE}" config
backup_target "${SESSION_CHECK}" session-check
backup_target "${UNIT_FILE}" service
backup_target "${START_TIMER}" start-timer
backup_target "${STOP_SERVICE}" stop-service
backup_target "${STOP_TIMER}" stop-timer
if [[ -L "${CURRENT_LINK}" ]]; then
    PRIOR_CURRENT_PRESENT=true
    PRIOR_CURRENT_TARGET=$(readlink -- "${CURRENT_LINK}")
elif [[ -e "${CURRENT_LINK}" ]]; then
    fail "${CURRENT_LINK} exists but is not a symlink"
fi
if systemctl is-active --quiet "${SERVICE_NAME}.service"; then PRIOR_SERVICE_ACTIVE=true; fi
PRIOR_SERVICE_ENABLED_STATE=$(systemctl is-enabled "${SERVICE_NAME}.service" 2>/dev/null || true)
PRIOR_START_TIMER_ENABLED_STATE=$(systemctl is-enabled "${SERVICE_NAME}-start.timer" 2>/dev/null || true)
PRIOR_STOP_TIMER_ENABLED_STATE=$(systemctl is-enabled "${SERVICE_NAME}-stop.timer" 2>/dev/null || true)
PRIOR_SERVICE_ENABLED_STATE=${PRIOR_SERVICE_ENABLED_STATE:-not-found}
PRIOR_START_TIMER_ENABLED_STATE=${PRIOR_START_TIMER_ENABLED_STATE:-not-found}
PRIOR_STOP_TIMER_ENABLED_STATE=${PRIOR_STOP_TIMER_ENABLED_STATE:-not-found}
for enabled_state in \
    "${PRIOR_SERVICE_ENABLED_STATE}" \
    "${PRIOR_START_TIMER_ENABLED_STATE}" \
    "${PRIOR_STOP_TIMER_ENABLED_STATE}"; do
    case "${enabled_state}" in
        enabled|enabled-runtime|disabled|static|indirect|masked|masked-runtime|not-found) ;;
        *) fail "unsupported existing systemd enablement state: ${enabled_state}" ;;
    esac
done
if systemctl is-active --quiet "${SERVICE_NAME}-start.timer"; then PRIOR_START_TIMER_ACTIVE=true; fi
if systemctl is-active --quiet "${SERVICE_NAME}-stop.timer"; then PRIOR_STOP_TIMER_ACTIVE=true; fi

atomic_install() {
    local source=$1
    local target=$2
    local mode=$3
    local owner=$4
    local group=$5
    local temporary="${target}.tmp.$$.$RANDOM"
    install -m "${mode}" -o "${owner}" -g "${group}" "${source}" "${temporary}"
    mv -Tf -- "${temporary}" "${target}"
}

log "activating the validated release"
ACTIVATION_STARTED=true
if unit_exists "${SERVICE_NAME}-start.timer"; then
    systemctl stop "${SERVICE_NAME}-start.timer"
fi
if unit_exists "${SERVICE_NAME}-stop.timer"; then
    systemctl stop "${SERVICE_NAME}-stop.timer"
fi
if unit_exists "${SERVICE_NAME}.service"; then
    systemctl stop "${SERVICE_NAME}.service"
    if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        fail "existing service did not stop; refusing to modify the active release"
    fi
fi

atomic_install "${CANONICAL_ENV}" "${ENV_FILE}" 0640 root "${SERVICE_GROUP}"
atomic_install "${CANDIDATE_CONFIG}" "${CONFIG_FILE}" 0640 root "${SERVICE_GROUP}"
atomic_install "${FINAL_DIR}/session-check" "${SESSION_CHECK}" 0755 root root
atomic_install "${FINAL_DIR}/${SERVICE_NAME}.service" "${UNIT_FILE}" 0644 root root
atomic_install "${FINAL_DIR}/${SERVICE_NAME}-start.timer" "${START_TIMER}" 0644 root root
atomic_install "${FINAL_DIR}/${SERVICE_NAME}-stop.service" "${STOP_SERVICE}" 0644 root root
atomic_install "${FINAL_DIR}/${SERVICE_NAME}-stop.timer" "${STOP_TIMER}" 0644 root root
next_link="${INSTALL_ROOT}/.current.${release_id}"
ln -s -- "${CANDIDATE_DIR}" "${next_link}"
mv -Tf -- "${next_link}" "${CURRENT_LINK}"

systemctl daemon-reload
systemd-analyze verify "${UNIT_FILE}" "${START_TIMER}" "${STOP_SERVICE}" "${STOP_TIMER}"
systemctl disable "${SERVICE_NAME}.service" >/dev/null
systemctl enable "${SERVICE_NAME}-start.timer" "${SERVICE_NAME}-stop.timer" >/dev/null

if "${START_NOW}"; then
    systemctl start "${SERVICE_NAME}-start.timer" "${SERVICE_NAME}-stop.timer"
    day=$(TZ=Asia/Kolkata date +%u)
    hhmm=$(TZ=Asia/Kolkata date +%H%M)
    if ((10#${day} <= 5 && 10#${hhmm} >= 910 && 10#${hhmm} < 1535)); then
        log "inside the weekday IST window; starting the live signal service"
        systemctl restart "${SERVICE_NAME}.service"
    else
        log "outside the weekday IST window; the timer will start it at 09:10 IST"
    fi
else
    systemctl stop "${SERVICE_NAME}.service" \
        "${SERVICE_NAME}-start.timer" "${SERVICE_NAME}-stop.timer" >/dev/null 2>&1 || true
    if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        fail "--no-start requested, but the live service is active"
    fi
    if systemctl is-active --quiet "${SERVICE_NAME}-start.timer" || \
       systemctl is-active --quiet "${SERVICE_NAME}-stop.timer"; then
        fail "--no-start requested, but an installation timer is active"
    fi
    log "--no-start honored: service and timers are inactive; timers remain enabled for next boot"
fi
ACTIVATION_COMPLETE=true

# Keep the current release plus the two newest rollback candidates.
mapfile -t installed_releases < <(
    find "${RELEASES_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr | cut -d' ' -f2-
)
kept=0
for release in "${installed_releases[@]}"; do
    preserve=false
    if [[ "${release}" == "${CANDIDATE_DIR}" ]]; then
        preserve=true
    elif "${PRIOR_CURRENT_PRESENT}" && [[ "${release}" == "${PRIOR_CURRENT_TARGET}" ]]; then
        preserve=true
    elif ((kept < 3)); then
        preserve=true
    fi
    if "${preserve}"; then
        ((kept += 1))
    else
        rm -rf -- "${release}"
    fi
done

log "installation complete: release ${release_id}"
printf '\nConfig:  %s\nSecrets: %s\nStatus:  sudo systemctl status %s\nLogs:    sudo journalctl -u %s -f\nTimers:  systemctl list-timers "%s-*"\n' \
    "${CONFIG_FILE}" "${ENV_FILE}" "${SERVICE_NAME}" "${SERVICE_NAME}" "${SERVICE_NAME}"
printf '\nIMPORTANT: this is a weekday IST schedule, not an NSE holiday calendar.\n'
printf 'IMPORTANT: this repository emits signals and models fills; it does not place or square off broker orders.\n'
