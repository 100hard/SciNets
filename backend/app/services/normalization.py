from __future__ import annotations

import re
from typing import Any, Dict, Optional

__all__ = [
    "infer_metric_from_text",
    "extract_measurement",
    "normalize_metric_name",
    "normalize_unit",
    "METRIC_SYNONYM_MAP",
]

# Baseline synonym map for common metrics across domains. Additional entries can
# be appended by loading configuration at runtime if needed.
_METRIC_SYNONYM_MAP: Dict[str, Dict[str, str]] = {
    "misclassification rate": {
        "normalized_metric": "Accuracy",
        "variant": "1 - error",
        "reason": "Misclassification rate implies complement of accuracy",
    },
    "error rate": {
        "normalized_metric": "Accuracy",
        "variant": "1 - error",
        "reason": "Error rate implies complement of accuracy",
    },
    "word error rate": {
        "normalized_metric": "WER",
        "variant": "WER",
        "reason": "Word error rate corresponds to WER metric",
    },
    "false positive rate": {
        "normalized_metric": "Specificity",
        "variant": "1 - FPR",
        "reason": "False positive rate implies specificity complement",
    },
    "precision": {
        "normalized_metric": "Precision",
        "variant": "precision",
        "reason": "Precision is already a canonical metric",
    },
    "recall": {
        "normalized_metric": "Recall",
        "variant": "recall",
        "reason": "Recall is already a canonical metric",
    },
    "f1": {
        "normalized_metric": "F1",
        "variant": "harmonic mean",
        "reason": "F1 refers to F1 score",
    },
    "f-score": {
        "normalized_metric": "F1",
        "variant": "harmonic mean",
        "reason": "F-score is the F1 score",
    },
    "bleu": {
        "normalized_metric": "BLEU",
        "variant": "BLEU",
        "reason": "BLEU is a canonical MT metric",
    },
    "rouge": {
        "normalized_metric": "ROUGE",
        "variant": "ROUGE",
        "reason": "ROUGE is a canonical summarisation metric",
    },
    "wer": {
        "normalized_metric": "WER",
        "variant": "WER",
        "reason": "WER is a canonical speech metric",
    },
    "accuracy": {
        "normalized_metric": "Accuracy",
        "variant": "accuracy",
        "reason": "Accuracy is canonical",
    },
    "auroc": {
        "normalized_metric": "AUROC",
        "variant": "roc",
        "reason": "Area under ROC curve",
    },
    "auc": {
        "normalized_metric": "AUROC",
        "variant": "roc",
        "reason": "AUC typically refers to AUROC",
    },
    "aupr": {
        "normalized_metric": "AUPR",
        "variant": "pr",
        "reason": "Area under precision/recall",
    },
    "psnr": {
        "normalized_metric": "PSNR",
        "variant": "psnr",
        "reason": "Peak signal-to-noise ratio",
    },
    "ssim": {
        "normalized_metric": "SSIM",
        "variant": "ssim",
        "reason": "Structural similarity index",
    },
}

_UNIT_NORMALISATION: Dict[str, str] = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "percentage point": "percentage_point",
    "percentage points": "percentage_point",
    "pp": "percentage_point",
    "points": "point",
    "point": "point",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "day": "day",
    "days": "day",
    "hz": "hz",
    "khz": "khz",
    "mhz": "mhz",
    "ghz": "ghz",
    "db": "dB",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "km": "km",
    "nm": "nm",
    "µm": "µm",
    "um": "µm",
    "mg": "mg",
    "g": "g",
    "kg": "kg",
    "ug": "µg",
    "µg": "µg",
    "ng": "ng",
    "mg/ml": "mg_per_ml",
    "ng/ml": "ng_per_ml",
    "mmol/l": "mmol_per_l",
    "mol/l": "mol_per_l",
    "ppm": "ppm",
}

_PERCENT_PATTERN = re.compile(
    r"(?P<value>-?\d{1,3}(?:[.,]\d+)?)(?:\s*)(?P<unit>%|percent|percentage|percentage points?|pp)",
    re.IGNORECASE,
)
_NUMBER_UNIT_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:[.,]\d+)?)(?:\s*)(?P<unit>ms|millisecond(?:s)?|s|sec(?:s)?|second(?:s)?|min(?:ute)?(?:s)?|h(?:ours?)?|day(?:s)?|hz|khz|mhz|ghz|db|mm|cm|m|km|nm|µm|um|mg|g|kg|ug|µg|ng|mg/ml|ng/ml|mmol/l|mol/l|ppm)",
    re.IGNORECASE,
)
_DECIMAL_PATTERN = re.compile(r"(?<!\d)(?P<value>-?\d+\.\d+)(?!\d)")


def infer_metric_from_text(object_text: str, evidence_text: str) -> Optional[Dict[str, Any]]:
    combined = f"{object_text} {evidence_text}".lower()
    for synonym, info in _METRIC_SYNONYM_MAP.items():
        if synonym in combined:
            payload: Dict[str, Any] = {
                "normalized_metric": info["normalized_metric"],
                "variant": info.get("variant"),
                "reason": info.get("reason"),
                "matched": synonym,
            }
            return {key: value for key, value in payload.items() if value}
    return None


def normalize_metric_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    for synonym, info in _METRIC_SYNONYM_MAP.items():
        if synonym in cleaned:
            return info["normalized_metric"]
    # If no explicit synonym match, fall back to capitalised default when the
    # token count is small to keep speculation bounded.
    if len(cleaned.split()) <= 4:
        return cleaned.title()
    return None


def normalize_unit(raw_unit: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"raw": raw_unit}
    if not raw_unit:
        return payload
    lowered = raw_unit.strip().lower()
    if not lowered:
        return payload
    canonical = _UNIT_NORMALISATION.get(lowered)
    if canonical:
        payload["normalized"] = canonical
    return payload


def _to_float(value: str) -> Optional[float]:
    if not value:
        return None
    normalized = value.replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def extract_measurement(
    text: Optional[str],
    *,
    metric_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = []
    for pattern, strategy in ((
        _PERCENT_PATTERN,
        "percent",
    ), (
        _NUMBER_UNIT_PATTERN,
        "number_with_unit",
    )):
        match = pattern.search(text)
        if not match:
            continue
        value = _to_float(match.group("value"))
        if value is None:
            continue
        unit_raw = match.group("unit")
        unit_payload = normalize_unit(unit_raw)
        canonical_unit = unit_payload.get("normalized")
        payload: Dict[str, Any] = {
            "value": value,
            "unit": canonical_unit or unit_raw,
            "unit_details": unit_payload,
            "source_text": match.group(0).strip(),
            "strategy": strategy,
        }
        if metric_hint:
            payload["metric_hint"] = metric_hint
        candidates.append(payload)
    if candidates:
        return candidates[0]

    if metric_hint:
        match = _DECIMAL_PATTERN.search(text)
        if match:
            value = _to_float(match.group("value"))
            if value is not None and value <= 1.0:
                return {
                    "value": value,
                    "unit": None,
                    "source_text": match.group(0).strip(),
                    "strategy": "decimal_ratio",
                    "metric_hint": metric_hint,
                }
    return None
