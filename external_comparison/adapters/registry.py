"""Native external adapter readiness checks.

The check is intentionally conservative: a method is available only when a
repository-local native adapter module exists. Controlled common-space
policies are never accepted as a substitute.
"""

from __future__ import annotations

import importlib.util

NATIVE_ADAPTER_MODULES = {
    "aflow": "external_comparison.adapters.native_aflow",
    "maas": "external_comparison.adapters.native_maas",
    "gdesigner": "external_comparison.adapters.native_gdesigner",
    "rpas": "external_comparison.adapters.native_rpas",
    "vanilla": "external_comparison.adapters.native_rpas",
    "rpas_no_selection": "external_comparison.adapters.native_rpas",
}


def native_adapter_status(method: str) -> dict[str, str | bool]:
    module_name = NATIVE_ADAPTER_MODULES.get(method)
    if module_name is None:
        return {"method": method, "available": False, "reason": "unknown_native_method"}
    available = importlib.util.find_spec(module_name) is not None
    return {
        "method": method,
        "available": available,
        "module": module_name,
        "reason": "ready" if available else "native_adapter_module_missing",
    }


def require_native_adapters(methods: tuple[str, ...] | list[str]) -> None:
    statuses = [native_adapter_status(method) for method in methods]
    missing = [status for status in statuses if not status["available"]]
    if missing:
        details = ", ".join(f"{item['method']} ({item['reason']})" for item in missing)
        raise RuntimeError(f"native external adapters are not ready: {details}")
