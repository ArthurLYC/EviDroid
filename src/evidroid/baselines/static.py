from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from androguard.misc import AnalyzeAPK

from evidroid.config import API_ALLOW_PREFIXES
from evidroid.extractors.androguard_extractor import _parse_invoked_api
from evidroid.features import normalize_feature_value
from evidroid.schemas import iter_evidence

try:
    from loguru import logger as loguru_logger

    loguru_logger.disable("androguard")
except Exception:
    pass

logging.getLogger("androguard").setLevel(logging.ERROR)


ANDROID_API_FAMILIES = {
    "android",
    "com.google",
    "dalvik",
    "java",
    "javax",
    "junit",
    "org.apache",
    "org.json",
    "org.w3c",
    "org.xml",
    "kotlin",
}

_DREBIN_NETWORK_RE = re.compile(
    r"(?i)\b(?:https?://|ftp://|www\.|(?:[a-z0-9-]+\.)+[a-z]{2,}|(?:\d{1,3}\.){3}\d{1,3})"
)
_DREBIN_NETWORK_TLDS = (
    ".com",
    ".net",
    ".org",
    ".cn",
    ".ru",
    ".info",
    ".biz",
    ".top",
    ".xyz",
)
_DREBIN_RESTRICTED_API_PREFIXES = (
    "Landroid/accounts/",
    "Landroid/app/admin/",
    "Landroid/bluetooth/",
    "Landroid/content/ContentResolver;",
    "Landroid/hardware/Camera;",
    "Landroid/location/",
    "Landroid/media/AudioRecord;",
    "Landroid/net/",
    "Landroid/provider/ContactsContract",
    "Landroid/provider/Settings$Secure;",
    "Landroid/telephony/",
    "Landroid/telephony/gsm/",
    "Landroid/telephony/cdma/",
)
_DREBIN_RESTRICTED_API_METHOD_MARKERS = (
    "->getDeviceId(",
    "->getSubscriberId(",
    "->getLine1Number(",
    "->getSimSerialNumber(",
    "->getLastKnownLocation(",
    "->requestLocationUpdates(",
    "->sendTextMessage(",
    "->sendMultipartTextMessage(",
    "->getAccounts(",
    "->query(",
)
_DREBIN_SUSPICIOUS_API_MARKERS = (
    "Ljava/lang/Runtime;->exec(",
    "Ljava/lang/System;->load(",
    "Ljava/lang/System;->loadLibrary(",
    "Ljava/lang/Class;->forName(",
    "Ljava/lang/reflect/",
    "Ldalvik/system/DexClassLoader;",
    "Ldalvik/system/PathClassLoader;",
    "Landroid/os/Debug;",
    "Ljavax/crypto/Cipher;",
)


def build_drebin_features(evidence_doc: dict[str, Any]) -> dict[str, float]:
    """Build a conservative original-style Drebin feature dictionary.

    The feature groups mirror Drebin's static categories that are available in
    EviDroid evidence: requested permissions, app components, restricted API
    calls, suspicious API calls, and network addresses. It intentionally avoids
    the previous enhanced setting that treated every API call and every
    suspicious string as a Drebin feature.
    """

    features: dict[str, float] = {}
    for item in iter_evidence(evidence_doc):
        view = item["view"]
        value = normalize_feature_value(view, item["value"])
        if view == "permission":
            features[f"drebin::requested_permission::{value}"] = 1.0
        elif view == "api":
            raw_api = str(item.get("value", ""))
            if _is_drebin_restricted_api(raw_api):
                features[f"drebin::restricted_api::{value}"] = 1.0
            if _is_drebin_suspicious_api(raw_api):
                features[f"drebin::suspicious_api::{value}"] = 1.0
        elif view == "component":
            component_type = item.get("detail", {}).get("component_type", "component")
            features[f"drebin::app_component::{component_type}"] = 1.0
            features[f"drebin::app_component_name::{value}"] = 1.0
        elif view == "string":
            if _is_drebin_network_address(str(item.get("value", ""))):
                features[f"drebin::network_address::{value}"] = 1.0
    return features


def build_droidapiminer_features(evidence_doc: dict[str, Any]) -> dict[str, float]:
    """Build DroidAPIMiner-style static API and permission features.

    DroidAPIMiner is a mined static-evidence baseline centered on API calls and
    requested permissions. This implementation keeps that evidence family
    boundary explicit instead of using EviDroid behavior labels or consistency
    signals.
    """

    features: dict[str, float] = {}
    for item in iter_evidence(evidence_doc):
        view = item["view"]
        if view not in {"api", "permission"}:
            continue
        value = normalize_feature_value(view, item["value"])
        if view == "api":
            features[f"droidapiminer::api::{value}"] = 1.0
        else:
            features[f"droidapiminer::permission::{value}"] = 1.0
    return features


def build_mamadroid_features_from_apk(
    apk_path: str | Path,
    abstraction: str = "package",
    max_calls: int = 100_000,
) -> dict[str, Any]:
    if abstraction not in {"family", "package"}:
        raise ValueError("MaMaDroid abstraction must be 'family' or 'package'.")

    apk_path = Path(apk_path)
    sequence: list[str] = []
    errors: list[str] = []
    try:
        _apk, dex_files, _analysis = AnalyzeAPK(str(apk_path))
        for dex in dex_files:
            for class_def in dex.get_classes():
                for method in class_def.get_methods():
                    if not hasattr(method, "get_code") or method.get_code() is None:
                        continue
                    try:
                        instructions = method.get_instructions()
                    except Exception:
                        continue
                    for instruction in instructions:
                        if not instruction.get_name().startswith("invoke-"):
                            continue
                        api = _parse_invoked_api(instruction.get_output())
                        if not api:
                            continue
                        sequence.append(abstract_api(api, abstraction=abstraction))
                        if len(sequence) >= max_calls:
                            break
                    if len(sequence) >= max_calls:
                        break
                if len(sequence) >= max_calls:
                    break
            if len(sequence) >= max_calls:
                break
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    return {
        "sample_id": apk_path.stem,
        "apk_path": str(apk_path),
        "abstraction": abstraction,
        "sequence_length": len(sequence),
        "features": markov_transition_features(sequence),
        "errors": errors,
    }


def build_mamadroid_features_from_evidence(
    evidence_doc: dict[str, Any],
    abstraction: str = "family",
    max_calls: int = 100_000,
) -> dict[str, float]:
    """Build MaMaDroid-style Markov transition features from API evidence.

    This follows MaMaDroid's key representation: abstract API calls to Android
    families or packages, then model transitions as a Markov chain. It uses the
    already extracted API evidence to avoid re-disassembling every APK.
    """

    if abstraction not in {"family", "package"}:
        raise ValueError("MaMaDroid abstraction must be 'family' or 'package'.")

    sequence: list[str] = []
    for item in iter_evidence(evidence_doc):
        if item["view"] != "api":
            continue
        sequence.append(abstract_api(item["value"], abstraction=abstraction))
        if len(sequence) >= max_calls:
            break
    return markov_transition_features(sequence)


def abstract_api(api: str, abstraction: str = "package") -> str:
    class_name = api.split("->", 1)[0]
    normalized = _normalize_dex_class(class_name)
    parts = normalized.split(".") if normalized else []
    if not parts:
        return "unknown"

    family = _api_family(parts)
    if abstraction == "family":
        return family
    if family == "self_defined":
        return family
    if family == "com.google" and len(parts) >= 3:
        return ".".join(parts[:3])
    if family in {"org.apache", "org.json", "org.w3c", "org.xml"} and len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return family


def markov_transition_features(sequence: list[str]) -> dict[str, float]:
    if len(sequence) < 2:
        return {}

    outgoing: dict[str, int] = defaultdict(int)
    transitions: Counter[tuple[str, str]] = Counter()
    for source, target in zip(sequence, sequence[1:]):
        outgoing[source] += 1
        transitions[(source, target)] += 1

    features: dict[str, float] = {}
    for (source, target), count in transitions.items():
        features[f"mamadroid::{source}->{target}"] = count / outgoing[source]
    return features


def _normalize_dex_class(class_name: str) -> str:
    class_name = class_name.strip()
    if class_name.startswith("["):
        class_name = class_name.lstrip("[")
    if class_name.startswith("L") and class_name.endswith(";"):
        class_name = class_name[1:-1]
    return class_name.replace("/", ".").replace("$", ".")


def _api_family(parts: list[str]) -> str:
    if parts[0] in {"android", "dalvik", "java", "javax", "junit", "kotlin"}:
        return parts[0]
    if len(parts) >= 2 and ".".join(parts[:2]) in ANDROID_API_FAMILIES:
        return ".".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "org":
        org_family = ".".join(parts[:2])
        if org_family in ANDROID_API_FAMILIES:
            return org_family
    return "self_defined"


def _is_drebin_restricted_api(value: str) -> bool:
    return value.startswith(_DREBIN_RESTRICTED_API_PREFIXES) or any(
        marker in value for marker in _DREBIN_RESTRICTED_API_METHOD_MARKERS
    )


def _is_drebin_suspicious_api(value: str) -> bool:
    return any(marker in value for marker in _DREBIN_SUSPICIOUS_API_MARKERS)


def _is_drebin_network_address(value: str) -> bool:
    lower = value.lower()
    has_ip_shape = "." in lower and any(ch.isdigit() for ch in lower)
    if (
        "://" not in lower
        and "www." not in lower
        and not has_ip_shape
        and not any(tld in lower for tld in _DREBIN_NETWORK_TLDS)
    ):
        return False
    return bool(_DREBIN_NETWORK_RE.search(value))
