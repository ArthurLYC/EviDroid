from __future__ import annotations

from pathlib import Path
from typing import Any

from evidroid.reporting import predict_detection
from evidroid.schemas import evidence_by_id


BEHAVIOR_RISK_TEXT = {
    "privacy_collection": "涉及设备标识、账号、联系人或电话状态等敏感信息，需确认是否符合业务用途和隐私合规要求。",
    "network_communication": "具备联网或上传能力，若与隐私收集、动态加载等行为组合出现，应重点关注数据外传或远程配置风险。",
    "dynamic_code_loading": "涉及 dex、类加载或反射相关线索，可能增加静态审计难度，也可能由正常框架或插件机制引入。",
    "sms_or_call_abuse": "涉及短信或拨号相关能力，可能带来资费损失、验证码拦截或未授权通信风险。",
    "boot_persistence": "涉及开机相关触发线索，可能提高应用驻留性，需要确认是否为正常后台服务需求。",
    "overlay_or_phishing": "涉及悬浮窗、无障碍或界面覆盖相关能力，需要关注钓鱼、诱导点击或界面劫持风险。",
    "command_execution": "涉及命令执行或系统二进制调用线索，需要重点排查是否存在越权访问、提权或环境探测行为。",
    "crypto_or_obfuscation": "涉及加密、摘要或编码能力，可能用于正常安全通信，也可能用于隐藏配置、载荷或通信内容。",
    "location_tracking": "涉及定位能力或位置字段，需结合联网行为确认是否存在位置轨迹外传风险。",
    "file_storage_access": "涉及外部存储或文件读写能力，可能用于正常缓存，也可能用于读取用户文件或落地载荷。",
    "package_installation": "涉及 APK 或安装包交互能力，需要关注下载、释放或诱导安装其他应用的风险。",
    "native_code_loading": "涉及原生库加载，可能由正常 SDK 引入，也可能用于反调试、加固或底层载荷执行。",
}


def generate_template_report(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    out_path: str | Path,
    prediction_doc: dict[str, Any] | None = None,
    max_behaviors: int = 6,
    max_evidence_per_behavior: int = 5,
) -> dict[str, Any]:
    markdown = build_template_report_markdown(
        evidence_doc=evidence_doc,
        behavior_doc=behavior_doc,
        prediction_doc=prediction_doc,
        max_behaviors=max_behaviors,
        max_evidence_per_behavior=max_evidence_per_behavior,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8", newline="\n")
    return {
        "sample_id": evidence_doc["sample_id"],
        "out_path": str(out_path),
        "prediction_label": (prediction_doc or {}).get("prediction_label"),
    }


def build_template_report_markdown(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    prediction_doc: dict[str, Any] | None = None,
    max_behaviors: int = 6,
    max_evidence_per_behavior: int = 5,
) -> str:
    prediction_doc = prediction_doc or {}
    behaviors = sorted(
        behavior_doc.get("behaviors", []),
        key=lambda row: float(row.get("consistency_score", 0.0)),
        reverse=True,
    )[:max_behaviors]
    risk_level = _risk_level(prediction_doc, behaviors)

    lines = [
        "# EviDroid Template Report",
        "",
        "## 样本信息",
        "",
        f"- Sample ID: `{evidence_doc.get('sample_id')}`",
        f"- Package: `{evidence_doc.get('package')}`",
        f"- SHA256: `{evidence_doc.get('sha256')}`",
    ]
    if prediction_doc:
        lines.extend(
            [
                f"- Detection result: `{prediction_doc.get('prediction_label')}`",
                f"- Malware score: `{_fmt_score(prediction_doc.get('malware_score'))}`",
            ]
        )
    lines.extend(
        [
            f"- Risk level: `{risk_level}`",
            "",
            "## 证据规模",
            "",
            "| View | Count |",
            "|---|---:|",
        ]
    )
    for view, count in sorted((evidence_doc.get("view_counts") or {}).items()):
        lines.append(f"| `{view}` | {count} |")

    lines.extend(["", "## 检测结论", ""])
    lines.extend(_conclusion_lines(prediction_doc, risk_level, behaviors))

    lines.extend(["", "## 关键行为发现", ""])
    if not behaviors:
        lines.append("- 未发现可报告的高层行为语义。")
    else:
        index = evidence_by_id(evidence_doc)
        for idx, behavior in enumerate(behaviors, start=1):
            label = str(behavior.get("label") or "unknown")
            lines.extend(
                [
                    f"### {idx}. {behavior.get('name') or label}",
                    "",
                    f"- Label: `{label}`",
                    f"- Consistency score: `{_fmt_score(behavior.get('consistency_score'))}`",
                    f"- Support by view: `{behavior.get('support_by_view', {})}`",
                    f"- Risk note: {BEHAVIOR_RISK_TEXT.get(label, behavior.get('description') or '该行为需要结合上下文进一步复核。')}",
                    "",
                    "| Evidence ID | View | Value |",
                    "|---|---|---|",
                ]
            )
            evidence_ids = behavior.get("evidence_ids", [])[:max_evidence_per_behavior]
            for evidence_id in evidence_ids:
                evidence = index.get(evidence_id)
                if not evidence:
                    continue
                lines.append(
                    f"| `{evidence_id}` | `{evidence.get('view')}` | `{_truncate(str(evidence.get('value', '')), 160)}` |"
                )
            lines.append("")

    lines.extend(["## 复核建议", ""])
    lines.extend(_recommendation_lines(prediction_doc, behaviors))
    return "\n".join(lines).rstrip() + "\n"


def prediction_for_template_report(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    model_path: str | Path | None,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
) -> dict[str, Any] | None:
    if not model_path:
        return None
    return predict_detection(
        evidence_doc=evidence_doc,
        behavior_doc=behavior_doc,
        model_path=model_path,
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
        static_profile=static_profile,
    )


def _risk_level(prediction_doc: dict[str, Any], behaviors: list[dict[str, Any]]) -> str:
    label = str(prediction_doc.get("prediction_label") or "").lower()
    score = prediction_doc.get("malware_score")
    if label == "malware":
        if isinstance(score, (int, float)) and score >= 0.8:
            return "high"
        return "medium"
    if label == "benign":
        high_consistency = any(float(item.get("consistency_score", 0.0)) >= 0.75 for item in behaviors)
        return "medium" if high_consistency else "low"
    if not behaviors:
        return "unknown"
    max_score = max(float(item.get("consistency_score", 0.0)) for item in behaviors)
    if max_score >= 0.75:
        return "medium"
    return "low"


def _conclusion_lines(
    prediction_doc: dict[str, Any],
    risk_level: str,
    behaviors: list[dict[str, Any]],
) -> list[str]:
    label = str(prediction_doc.get("prediction_label") or "unknown")
    score = prediction_doc.get("malware_score")
    if label in {"benign", "malware"}:
        return [
            f"- 分类器判定结果为 `{label}`，恶意分数为 `{_fmt_score(score)}`。",
            f"- 报告风险等级为 `{risk_level}`，共纳入 {len(behaviors)} 条高层行为发现。",
            "- 本报告基于静态证据、分类器输出和已验证行为记录生成，不代表动态行为已实际发生。",
        ]
    return [
        "- 当前未提供分类器预测结果，报告仅总结静态证据和行为发现。",
        f"- 报告风险等级为 `{risk_level}`，共纳入 {len(behaviors)} 条高层行为发现。",
        "- 本报告基于静态证据和已验证行为记录生成，不代表动态行为已实际发生。",
    ]


def _recommendation_lines(prediction_doc: dict[str, Any], behaviors: list[dict[str, Any]]) -> list[str]:
    label = str(prediction_doc.get("prediction_label") or "").lower()
    behavior_labels = {str(item.get("label")) for item in behaviors}
    lines: list[str] = []
    if label == "malware":
        lines.append("- 建议隔离样本并结合动态沙箱确认网络通信、载荷释放和敏感数据访问行为。")
    elif label == "benign":
        lines.append("- 当前分类器判定为良性，建议仅对高一致性行为进行低优先级复核。")
    else:
        lines.append("- 建议结合分类器预测或人工分析结果进一步确定处置优先级。")

    if "dynamic_code_loading" in behavior_labels or "native_code_loading" in behavior_labels:
        lines.append("- 若存在动态加载或原生库加载，建议进一步检查释放文件、加载路径和运行时调用链。")
    if "privacy_collection" in behavior_labels and "network_communication" in behavior_labels:
        lines.append("- 若隐私收集与联网行为同时出现，建议重点复核是否存在敏感数据外传链路。")
    if "sms_or_call_abuse" in behavior_labels:
        lines.append("- 若存在短信或拨号相关能力，建议结合权限使用路径确认是否存在资费或验证码风险。")
    return lines


def _fmt_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _truncate(value: str, max_length: int) -> str:
    value = value.replace("\n", "\\n").replace("|", "\\|")
    return value if len(value) <= max_length else value[: max_length - 3] + "..."
