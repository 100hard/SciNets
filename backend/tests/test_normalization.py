import pytest

from app.services.normalization import (
    extract_measurement,
    infer_metric_from_text,
    normalize_metric_name,
    normalize_unit,
)


def test_infer_metric_from_text_matches_synonym():
    result = infer_metric_from_text("Misclassification", "The misclassification rate drops significantly")
    assert result is not None
    assert result["normalized_metric"] == "Accuracy"
    assert result["matched"] == "misclassification rate"


def test_extract_measurement_percent():
    measurement = extract_measurement("Accuracy improves to 92.5% on CIFAR-10")
    assert measurement is not None
    assert pytest.approx(measurement["value"], rel=1e-6) == 92.5
    assert measurement["unit"] == "%"
    assert measurement["unit_details"]["normalized"] == "%"


def test_extract_measurement_decimal_ratio_with_hint():
    measurement = extract_measurement("F1 rises to 0.87 on the dev set", metric_hint="F1")
    assert measurement is not None
    assert pytest.approx(measurement["value"], rel=1e-6) == 0.87
    assert measurement["strategy"] == "decimal_ratio"


def test_normalize_metric_name_fallback():
    assert normalize_metric_name("macro f1") == "Macro F1"


def test_normalize_unit_handles_synonyms():
    payload = normalize_unit("milliseconds")
    assert payload["normalized"] == "ms"
