from __future__ import annotations

import hashlib
import logging
import re
import string
from pathlib import Path
from typing import Any

from androguard.misc import AnalyzeAPK

from evidroid.config import API_ALLOW_PREFIXES, SUSPICIOUS_STRING_MARKERS, VIEW_PREFIXES
from evidroid.schemas import Evidence

try:
    from loguru import logger as loguru_logger

    loguru_logger.disable("androguard")
except Exception:
    pass

logging.getLogger("androguard").setLevel(logging.ERROR)

INVOKE_API_RE = re.compile(r"(L[^;]+;)->([^\s(]+)\((.*?)\)(\S+)")


class AndroguardEvidenceExtractor:
    def __init__(
        self,
        max_apis: int = 2500,
        max_strings: int = 800,
        include_internal_apis: bool = False,
    ) -> None:
        self.max_apis = max_apis
        self.max_strings = max_strings
        self.include_internal_apis = include_internal_apis

    def extract(self, apk_path: str | Path, label: str | None = None) -> dict[str, Any]:
        apk_path = Path(apk_path)
        evidence: list[Evidence] = []
        errors: list[str] = []
        package = None

        try:
            apk, dex_files, _analysis = AnalyzeAPK(str(apk_path))
            package = apk.get_package()
        except Exception as exc:
            errors.append(f"AnalyzeAPK failed: {type(exc).__name__}: {exc}")
            dex_files = []
            apk = None

        if apk is not None:
            for name, extractor in (
                ("permission", lambda: self._permission_evidence(apk)),
                ("component", lambda: self._component_evidence(apk)),
                ("api", lambda: self._api_evidence(dex_files)),
                ("string", lambda: self._string_evidence(dex_files)),
            ):
                try:
                    evidence.extend(extractor())
                except Exception as exc:
                    errors.append(f"{name} extraction failed: {type(exc).__name__}: {exc}")

        evidence = self._assign_ids(evidence)
        return {
            "sample_id": apk_path.stem,
            "apk_path": str(apk_path),
            "sha256": sha256_file(apk_path),
            "label": label,
            "package": package,
            "evidence": [item.to_dict() for item in evidence],
            "view_counts": _view_counts(evidence),
            "errors": errors,
        }

    def _permission_evidence(self, apk: Any) -> list[Evidence]:
        rows: list[Evidence] = []
        for permission in sorted(set(apk.get_permissions() or [])):
            rows.append(
                Evidence(
                    id="",
                    view="permission",
                    value=permission,
                    detail={"permission": permission.split(".")[-1]},
                )
            )
        return rows

    def _component_evidence(self, apk: Any) -> list[Evidence]:
        rows: list[Evidence] = []
        component_getters = {
            "activity": apk.get_activities,
            "service": apk.get_services,
            "receiver": apk.get_receivers,
            "provider": apk.get_providers,
        }
        for component_type, getter in component_getters.items():
            try:
                values = getter() or []
            except Exception:
                values = []
            for value in sorted(set(values)):
                rows.append(
                    Evidence(
                        id="",
                        view="component",
                        value=f"{component_type}:{value}",
                        detail={"component_type": component_type, "name": value},
                    )
                )
        return rows

    def _api_evidence(self, dex_files: list[Any]) -> list[Evidence]:
        apis: set[str] = set()
        for dex in dex_files:
            for class_def in dex.get_classes():
                methods = class_def.get_methods()
                for method in methods:
                    if not hasattr(method, "get_code"):
                        continue
                    code = method.get_code()
                    if code is None:
                        continue
                    try:
                        instructions = method.get_instructions()
                    except Exception:
                        continue
                    for instruction in instructions:
                        name = instruction.get_name()
                        if not name.startswith("invoke-"):
                            continue
                        api = _parse_invoked_api(instruction.get_output())
                        if not api:
                            continue
                        if self.include_internal_apis or api.startswith(API_ALLOW_PREFIXES):
                            apis.add(api)
                        if len(apis) >= self.max_apis:
                            break
                    if len(apis) >= self.max_apis:
                        break
                if len(apis) >= self.max_apis:
                    break
            if len(apis) >= self.max_apis:
                break

        rows: list[Evidence] = []
        for api in sorted(apis):
            class_name, method_name = api.split("->", 1)
            rows.append(
                Evidence(
                    id="",
                    view="api",
                    value=api,
                    detail={"class": class_name, "method": method_name},
                )
            )
        return rows

    def _string_evidence(self, dex_files: list[Any]) -> list[Evidence]:
        strings: set[str] = set()
        for dex in dex_files:
            try:
                raw_strings = dex.get_strings()
            except Exception:
                raw_strings = []
            for item in raw_strings:
                text = str(item)
                if _keep_string(text):
                    strings.add(text)

        ordered = sorted(strings, key=_string_priority)[: self.max_strings]
        return [
            Evidence(id="", view="string", value=value, detail={"length": len(value)})
            for value in ordered
        ]

    def _assign_ids(self, evidence: list[Evidence]) -> list[Evidence]:
        counts = {view: 0 for view in VIEW_PREFIXES}
        assigned: list[Evidence] = []
        for item in evidence:
            counts[item.view] += 1
            prefix = VIEW_PREFIXES[item.view]
            assigned.append(
                Evidence(
                    id=f"{prefix}_{counts[item.view]:04d}",
                    view=item.view,
                    value=item.value,
                    detail=item.detail,
                )
            )
        return assigned


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_invoked_api(output: str) -> str | None:
    match = INVOKE_API_RE.search(output)
    if not match:
        return None
    class_name, method_name, descriptor_args, descriptor_ret = match.groups()
    return f"{class_name}->{method_name}({descriptor_args}){descriptor_ret}"


def _keep_string(text: str) -> bool:
    if not 4 <= len(text) <= 180:
        return False
    if "\x00" in text:
        return False
    printable = set(string.printable)
    ascii_ratio = sum(1 for char in text if char in printable) / max(1, len(text))
    if ascii_ratio < 0.75:
        return False
    if text.count(" ") > len(text) * 0.6:
        return False
    return True


def _string_priority(text: str) -> tuple[int, int, str]:
    lower = text.lower()
    suspicious = any(marker.lower() in lower for marker in SUSPICIOUS_STRING_MARKERS)
    return (0 if suspicious else 1, len(text), lower)


def _view_counts(evidence: list[Evidence]) -> dict[str, int]:
    counts = {view: 0 for view in VIEW_PREFIXES}
    for item in evidence:
        counts[item.view] = counts.get(item.view, 0) + 1
    return counts
