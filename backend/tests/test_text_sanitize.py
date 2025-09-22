from app.utils.text_sanitize import sanitize_text


def test_sanitize_text_removes_null_bytes() -> None:
    raw = "Hello\x00World\x00"
    cleaned = sanitize_text(raw)
    assert cleaned == "HelloWorld"


def test_sanitize_text_preserves_newlines_and_tabs() -> None:
    raw = "Line1\n\tIndented\nLine3"
    cleaned = sanitize_text(raw)
    assert cleaned == "Line1\n\tIndented\nLine3"


def test_sanitize_text_returns_none_when_empty() -> None:
    raw = "\x00\x00\u0003"
    cleaned = sanitize_text(raw)
    assert cleaned is None
