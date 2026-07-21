from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib

from evidroid.features import build_evidroid_feature_dict
from evidroid.schemas import evidence_by_id

EVIDENCE_ID_RE = re.compile(r"\b(?:PERM|API|COMP|STR)_\d{4}\b")

MALWARE_RATIONALE_BY_LABEL = {
    "privacy_collection": "涉及设备标识、账号、联系人或电话状态等敏感信息，若与网络通信组合出现，可能形成数据收集链路。",
    "network_communication": "具备联网或上传能力，可支撑远程控制、配置拉取或敏感数据外传。",
    "dynamic_code_loading": "动态加载 dex、类或反射调用可隐藏真实载荷，使静态审计难以完整覆盖。",
    "sms_or_call_abuse": "短信或拨号能力可能造成资费损失、验证码拦截或未授权通信。",
    "boot_persistence": "开机自启动能力可提高驻留性，使样本在重启后继续运行。",
    "overlay_or_phishing": "悬浮窗或无障碍相关能力可被用于界面覆盖、钓鱼或诱导操作。",
    "command_execution": "命令执行能力可能访问系统信息、调用特权二进制或配合提权链路。",
    "crypto_or_obfuscation": "加密、摘要或编码能力可用于隐藏配置、保护载荷或加密通信内容。",
    "location_tracking": "位置访问能力可能导致用户轨迹泄露，需结合联网行为重点复核。",
    "file_storage_access": "外部存储访问可用于读取用户文件、落地载荷或保存中间数据。",
    "package_installation": "安装包交互能力可能用于下载、释放或诱导安装其他 APK。",
    "native_code_loading": "原生库加载可绕过部分 Java 层分析，常用于反调试、加固或执行底层逻辑。",
}


def generate_llm_analyst_report(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    out_path: str | Path,
    llm_config: dict[str, Any],
    prediction_doc: dict[str, Any] | None = None,
    language: str = "zh-CN",
    max_behaviors: int = 4,
    max_evidence_per_behavior: int = 4,
) -> dict[str, Any]:
    from openai import OpenAI

    api_key = llm_config.get("api_key")
    if not api_key:
        raise ValueError("DeepSeek API key is empty. Fill configs/deepseek.json llm.api_key or set DEEPSEEK_API_KEY.")

    verdict = _prediction_label(prediction_doc)
    if verdict == "malware":
        max_behaviors = max(max_behaviors, 5)
        max_evidence_per_behavior = max(max_evidence_per_behavior, 5)
    context = build_report_context(
        evidence_doc,
        behavior_doc,
        prediction_doc=prediction_doc,
        max_behaviors=max_behaviors,
        max_evidence_per_behavior=max_evidence_per_behavior,
    )
    client = OpenAI(api_key=api_key, base_url=llm_config.get("base_url"))
    request: dict[str, Any] = {
        "model": str(llm_config.get("model", "deepseek-v4-flash")),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是移动安全分析报告撰写助手。只根据用户提供的 JSON 编写报告。"
                    "不要改变 classifier_result 中的检测结论；不要把静态证据描述为已发生的动态行为；"
                    "所有技术判断必须引用输入中存在的 evidence_id。没有证据时写明证据不足。"
                    "输出简洁 Markdown，不要输出 JSON，不要编造证据 ID。"
                    "如果检测结果是 benign，报告应以复核和低优先级处置为主，不要写成恶意处置报告。"
                    "如果检测结果是 malware，必须用 malware_rationale 解释为什么这些证据支持恶意判断。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "language": language,
                        "task": _report_task(context, language),
                        "context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": float(llm_config.get("report_temperature", llm_config.get("temperature", 0.2)) or 0),
        "max_tokens": int(llm_config.get("report_max_tokens") or 1200),
        "stream": False,
    }
    thinking_type = _thinking_type(llm_config.get("thinking"))
    if thinking_type:
        request["extra_body"] = {"thinking": {"type": thinking_type}}
    response = client.chat.completions.create(**request)
    body = (response.choices[0].message.content or "").strip()
    unknown_ids = unknown_evidence_ids(body, evidence_doc)
    markdown = _analyst_report_header(context) + "\n\n" + body.strip() + "\n"
    if unknown_ids:
        markdown += "\n## Evidence ID Validation\n\n"
        markdown += "The LLM report referenced unknown evidence IDs: "
        markdown += ", ".join(f"`{item}`" for item in unknown_ids)
        markdown += "\n"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8", newline="\n")
    usage = getattr(response, "usage", None)
    result = {
        "sample_id": evidence_doc["sample_id"],
        "out_path": str(out_path),
        "unknown_evidence_ids": unknown_ids,
    }
    if usage is not None:
        result["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return result


def build_report_context(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    prediction_doc: dict[str, Any] | None = None,
    max_behaviors: int = 4,
    max_evidence_per_behavior: int = 4,
) -> dict[str, Any]:
    index = evidence_by_id(evidence_doc)
    behaviors = sorted(
        behavior_doc.get("behaviors", []),
        key=lambda row: float(row.get("consistency_score", 0.0)),
        reverse=True,
    )[:max_behaviors]
    compact_behaviors: list[dict[str, Any]] = []
    for item in behaviors:
        evidence_items = []
        for evidence_id in item.get("evidence_ids", [])[:max_evidence_per_behavior]:
            evidence = index.get(evidence_id)
            if not evidence:
                continue
            evidence_items.append(
                {
                    "id": evidence_id,
                    "view": evidence.get("view"),
                    "value": _truncate(str(evidence.get("value", "")), 220),
                }
            )
        compact_behaviors.append(
            {
                "label": item.get("label"),
                "name": item.get("name"),
                "description": item.get("description"),
                "consistency_score": item.get("consistency_score"),
                "support_by_view": item.get("support_by_view", {}),
                "evidence": evidence_items,
            }
        )
    return {
        "sample": {
            "sample_id": evidence_doc.get("sample_id"),
            "package": evidence_doc.get("package"),
            "sha256": evidence_doc.get("sha256"),
            "view_counts": evidence_doc.get("view_counts", {}),
            "apk_path": evidence_doc.get("apk_path"),
        },
        "classifier_result": prediction_doc or {},
        "report_profile": _report_profile(prediction_doc),
        "behavior_findings": compact_behaviors,
        "malware_rationale": _malware_rationale_context(compact_behaviors, prediction_doc),
        "report_constraints": {
            "do_not_use_ground_truth_label": True,
            "cite_only_evidence_ids_in_behavior_findings": True,
            "static_analysis_only": True,
        },
    }


def predict_detection(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    model_path: str | Path,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
) -> dict[str, Any]:
    model_path = Path(model_path)
    model = joblib.load(model_path)
    features = build_evidroid_feature_dict(
        evidence_doc,
        behavior_doc,
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
        static_profile=static_profile,
    )
    prediction = int(model.predict([features])[0])
    score, score_type = _malware_score(model, [features], prediction)
    return {
        "prediction": prediction,
        "prediction_label": "malware" if prediction == 1 else "benign",
        "malware_score": score,
        "score_type": score_type,
        "model_path": str(model_path),
        "feature_count": len(features),
    }


def unknown_evidence_ids(markdown: str, evidence_doc: dict[str, Any]) -> list[str]:
    allowed = set(evidence_by_id(evidence_doc))
    return sorted({item for item in EVIDENCE_ID_RE.findall(markdown) if item not in allowed})


def _malware_score(model: Any, x_rows: list[dict[str, float]], prediction: int) -> tuple[float | None, str | None]:
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    transformed = model[:-1].transform(x_rows) if hasattr(model, "steps") else x_rows
    if hasattr(classifier, "predict_proba"):
        return float(classifier.predict_proba(transformed)[0][1]), "probability"
    if hasattr(classifier, "decision_function"):
        return float(classifier.decision_function(transformed)[0]), "decision_function"
    return float(prediction), "prediction"


def _analyst_report_header(context: dict[str, Any]) -> str:
    sample = context["sample"]
    prediction = context.get("classifier_result", {})
    lines = [
        "# EviDroid Analyst Report",
        "",
        f"- Sample ID: `{sample.get('sample_id')}`",
        f"- Package: `{sample.get('package')}`",
        f"- SHA256: `{sample.get('sha256')}`",
    ]
    if prediction:
        lines.append(f"- Detection result: `{prediction.get('prediction_label')}`")
        if prediction.get("malware_score") is not None:
            lines.append(f"- Malware score: `{float(prediction['malware_score']):.4f}`")
    return "\n".join(lines)


def _prediction_label(prediction_doc: dict[str, Any] | None) -> str:
    if not prediction_doc:
        return "unknown"
    return str(prediction_doc.get("prediction_label") or "unknown").lower()


def _report_profile(prediction_doc: dict[str, Any] | None) -> dict[str, Any]:
    verdict = _prediction_label(prediction_doc)
    if verdict == "benign":
        return {
            "template": "benign_review",
            "purpose": "brief low-priority review note; avoid malware-style claims unless evidence is strong",
        }
    if verdict == "malware":
        return {
            "template": "malware_triage",
            "purpose": "brief response-oriented malware triage note",
        }
    return {
        "template": "unknown_triage",
        "purpose": "brief evidence summary without final verdict",
    }


def _report_task(context: dict[str, Any], language: str) -> str:
    verdict = str(context.get("classifier_result", {}).get("prediction_label") or "unknown").lower()
    if verdict == "malware":
        return (
            f"Write in {language}. Write a concise but slightly detailed analyst-facing malware triage report. "
            "Keep it under 950 Chinese characters. "
            "Use sections: 检测结论, 恶意判定依据, 关键证据, 处置建议. "
            "恶意判定依据 must explain why the behavior combination is suspicious in analyst-friendly language. "
            "Use at most 4 bullets in 关键证据, at most 3 bullets in other sections. "
            "Keep every claim tied to evidence IDs."
        )
    if verdict == "benign":
        return (
            f"Write in {language}. Write a brief benign APK review note. "
            "Keep it under 450 Chinese characters. "
            "Use sections: 检测结论, 复核关注点, 建议. "
            "Use at most 3 bullets per section. "
            "Do not phrase review points as confirmed malicious behavior."
        )
    return (
        f"Write in {language}. Write a concise APK triage report under 550 Chinese characters. "
        "Use sections: 检测结论, 关键发现, 建议. "
        "Use at most 3 bullets per section."
    )


def _malware_rationale_context(
    behaviors: list[dict[str, Any]],
    prediction_doc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if _prediction_label(prediction_doc) != "malware":
        return []
    rationale = []
    for item in behaviors[:4]:
        label = str(item.get("label") or "")
        rationale.append(
            {
                "label": label,
                "name": item.get("name"),
                "why_it_matters": MALWARE_RATIONALE_BY_LABEL.get(label)
                or item.get("description")
                or "该行为会提高样本风险，需要结合运行时行为验证。",
                "evidence_ids": [evidence.get("id") for evidence in item.get("evidence", []) if evidence.get("id")],
            }
        )
    return rationale
def _truncate(value: str, max_length: int) -> str:
    return value if len(value) <= max_length else value[: max_length - 3] + "..."


def _thinking_type(value: dict[str, Any] | str | bool | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"enabled", "disabled"}:
            return lowered
        if lowered in {"true", "yes", "on"}:
            return "enabled"
        if lowered in {"false", "no", "off"}:
            return "disabled"
    if isinstance(value, dict):
        raw_type = str(value.get("type", "")).strip().lower()
        if raw_type in {"enabled", "disabled"}:
            return raw_type
    raise ValueError(f"Unsupported thinking config: {value!r}")
