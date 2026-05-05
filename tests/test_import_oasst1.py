from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_importer():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "import_oasst1.py"
    spec = importlib.util.spec_from_file_location("import_oasst1", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(role: str, text: str, *, parent_id: str | None = None, quality: float = 0.8):
    return {
        "role": role,
        "text": text,
        "parent_id": parent_id,
        "created_date": "2023-01-01T00:00:00",
        "labels": {
            "quality": {"value": quality},
            "toxicity": {"value": 0.0},
        },
    }


def test_chat_compact_filter_keeps_short_prose_windows(tmp_path):
    mod = _load_importer()
    messages = {
        "u1": _row("prompter", "Explain rainbows simply."),
        "a1": _row(
            "assistant",
            "Rainbows happen when sunlight bends, reflects, and splits into colors inside raindrops.",
            parent_id="u1",
        ),
        "u2": _row("prompter", "Can you make that even shorter?", parent_id="a1"),
        "a2": _row(
            "assistant",
            "Light hits raindrops and comes back out as a band of colors.",
            parent_id="u2",
        ),
    }

    stats = mod._build_chat_compact_corpus(
        messages=messages,
        output=tmp_path / "chat.txt",
        window_messages=4,
        min_user_chars=8,
        max_user_chars=280,
        min_assistant_chars=48,
        max_assistant_chars=900,
        min_user_quality=0.2,
        min_assistant_quality=0.45,
        max_toxicity=0.5,
    )

    text = (tmp_path / "chat.txt").read_text(encoding="utf-8")
    assert stats["transcripts"] == 2
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "Explain rainbows simply." in text
    assert "Can you make that even shorter?" in text


def test_chat_compact_filter_rejects_code_and_ai_disclaimer(tmp_path):
    mod = _load_importer()
    messages = {
        "u1": _row("prompter", "Write a Python function."),
        "a1": _row(
            "assistant",
            "```python\ndef add(a, b):\n    total = a + b\n    return total\n```",
            parent_id="u1",
        ),
        "u2": _row("prompter", "Who are you?"),
        "a2": _row(
            "assistant",
            "As an AI language model, I can help with general questions and drafting.",
            parent_id="u2",
        ),
    }

    stats = mod._build_chat_compact_corpus(
        messages=messages,
        output=tmp_path / "chat.txt",
        window_messages=4,
        min_user_chars=8,
        max_user_chars=280,
        min_assistant_chars=48,
        max_assistant_chars=900,
        min_user_quality=0.2,
        min_assistant_quality=0.45,
        max_toxicity=0.5,
    )

    assert stats["transcripts"] == 0
    assert "```" in stats["rejected"]
    assert r"\bas an ai\b" in stats["rejected"]


def test_chat_creative_profile_prefers_creative_or_supportive_prompts(tmp_path):
    mod = _load_importer()
    messages = {
        "u1": _row("prompter", "Write me a short bedtime story about a moon rabbit."),
        "a1": _row(
            "assistant",
            "Once there was a moon rabbit who stitched silver dreams into the night sky for sleepy children below.",
            parent_id="u1",
        ),
        "u2": _row("prompter", "What GPU should I buy for Open Assistant?"),
        "a2": _row(
            "assistant",
            "A midrange GPU with enough memory will usually be the practical starting point for local experimentation.",
            parent_id="u2",
        ),
    }

    stats = mod._build_chat_compact_corpus(
        messages=messages,
        output=tmp_path / "chat.txt",
        window_messages=4,
        min_user_chars=8,
        max_user_chars=280,
        min_assistant_chars=48,
        max_assistant_chars=900,
        min_user_quality=0.2,
        min_assistant_quality=0.45,
        max_toxicity=0.5,
        include_patterns=mod.CHAT_CREATIVE_INCLUDE_PATTERNS,
        exclude_patterns=mod.CHAT_CREATIVE_EXCLUDE_PATTERNS,
    )

    text = (tmp_path / "chat.txt").read_text(encoding="utf-8")
    assert stats["transcripts"] == 1
    assert "moon rabbit" in text
    assert "Open Assistant" not in text
    assert "profile_exclude" in stats["rejected"]
