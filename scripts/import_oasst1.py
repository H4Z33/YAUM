"""Build a YAUM-ready chat corpus from OpenAssistant OASST1.

The trainer consumes plain text, so this script converts OASST1's message
tree JSONL into transcript blocks:

    <|user|>
    ...
    <|assistant|>
    ...
    <|end|>

It uses only the already-installed ``huggingface_hub`` package and the
standard library.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


READY_MESSAGES = "2023-04-12_oasst_ready.messages.jsonl.gz"
ROLE_LABELS = {
    "prompter": "<|user|>",
    "assistant": "<|assistant|>",
}
CHAT_COMPACT_REJECTION_TOKENS = (
    "import ",
    "def ",
    "class ",
    "SELECT ",
    "CREATE TABLE",
    "</",
    "/>",
    "{",
    "}",
)
CHAT_COMPACT_REJECTION_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"`"),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"www\.", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\blanguage model\b", re.IGNORECASE),
)


def _normalise_charset(text: str, charset: str) -> str:
    if charset == "raw":
        return text
    if charset != "ascii":
        raise ValueError(f"unknown charset mode: {charset}")
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", "ignore").decode("ascii")


def _clean_text(text: str, charset: str) -> str:
    text = _normalise_charset(text, charset)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _label_value(row: dict, name: str, default: float = 0.0) -> float:
    labels = row.get("labels") or {}
    metric = labels.get(name) or {}
    value = metric.get("value", default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_messages(path: str, lang: str, charset: str) -> dict[str, dict]:
    messages: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("deleted"):
                continue
            if row.get("lang") != lang:
                continue
            role = row.get("role")
            text = _clean_text(row.get("text") or "", charset)
            message_id = row.get("message_id")
            if role not in ROLE_LABELS or not text or not message_id:
                continue
            row["text"] = text
            messages[message_id] = row
    return messages


def _children_by_parent(messages: dict[str, dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for message_id, row in messages.items():
        parent_id = row.get("parent_id")
        if parent_id and parent_id in messages:
            children[parent_id].append(message_id)

    def sort_key(message_id: str):
        row = messages[message_id]
        rank = row.get("rank")
        return (
            rank is None,
            rank if rank is not None else 10**9,
            row.get("created_date") or "",
            message_id,
        )

    for ids in children.values():
        ids.sort(key=sort_key)
    return children


def _roots(messages: dict[str, dict]) -> list[str]:
    roots = [
        message_id
        for message_id, row in messages.items()
        if not row.get("parent_id") or row.get("parent_id") not in messages
    ]
    roots.sort(key=lambda mid: (messages[mid].get("created_date") or "", mid))
    return roots


def _paths_from(
    message_id: str,
    children: dict[str, list[str]],
    prefix: list[str],
    *,
    all_paths: bool,
) -> Iterable[list[str]]:
    path = [*prefix, message_id]
    child_ids = children.get(message_id, [])
    if not child_ids:
        yield path
        return
    if not all_paths:
        child_ids = child_ids[:1]
    for child_id in child_ids:
        yield from _paths_from(child_id, children, path, all_paths=all_paths)


def _format_path(path: list[str], messages: dict[str, dict]) -> str:
    parts: list[str] = []
    last_role = None
    for message_id in path:
        row = messages[message_id]
        role = row["role"]
        if role == last_role:
            parts.append(row["text"])
        else:
            parts.append(f"{ROLE_LABELS[role]}\n{row['text']}")
        last_role = role
    return "\n\n".join(parts) + "\n\n<|end|>\n"


def _iter_assistant_windows(
    messages: dict[str, dict],
    *,
    window_messages: int,
) -> Iterable[list[str]]:
    assistant_ids = [
        message_id
        for message_id, row in messages.items()
        if row.get("role") == "assistant"
    ]
    assistant_ids.sort(key=lambda mid: (messages[mid].get("created_date") or "", mid))
    for message_id in assistant_ids:
        path: list[str] = []
        current = message_id
        while current in messages and len(path) < window_messages:
            path.append(current)
            current = messages[current].get("parent_id")
        yield list(reversed(path))


def _is_alternating_chat_path(path: list[str], messages: dict[str, dict]) -> bool:
    if len(path) < 2:
        return False
    if messages[path[0]].get("role") != "prompter":
        return False
    if messages[path[-1]].get("role") != "assistant":
        return False
    for idx, message_id in enumerate(path):
        expected = "prompter" if idx % 2 == 0 else "assistant"
        if messages[message_id].get("role") != expected:
            return False
    return True


def _chat_compact_reject_reason(text: str) -> str | None:
    if text.count("\n") > 8:
        return "too_many_newlines"
    for pattern in CHAT_COMPACT_REJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    for token in CHAT_COMPACT_REJECTION_TOKENS:
        if token in text:
            return token
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(len(text), 1) < 0.5:
        return "low_alpha_ratio"
    return None


def _build_chat_compact_corpus(
    *,
    messages: dict[str, dict],
    output: Path,
    window_messages: int,
    min_user_chars: int,
    max_user_chars: int,
    min_assistant_chars: int,
    max_assistant_chars: int,
    min_user_quality: float,
    min_assistant_quality: float,
    max_toxicity: float,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_blocks: set[str] = set()
    n_paths = 0
    n_chars = 0
    length_hist: dict[int, int] = defaultdict(int)
    rejected: dict[str, int] = defaultdict(int)

    with output.open("w", encoding="utf-8", newline="\n") as out:
        for path in _iter_assistant_windows(messages, window_messages=window_messages):
            if not _is_alternating_chat_path(path, messages):
                rejected["bad_path_shape"] += 1
                continue

            passed = True
            for message_id in path:
                row = messages[message_id]
                text = row["text"]
                role = row["role"]
                text_len = len(text)
                if role == "prompter":
                    if text_len < min_user_chars or text_len > max_user_chars:
                        rejected["user_len"] += 1
                        passed = False
                        break
                    if _label_value(row, "quality") < min_user_quality:
                        rejected["user_quality"] += 1
                        passed = False
                        break
                else:
                    if text_len < min_assistant_chars or text_len > max_assistant_chars:
                        rejected["assistant_len"] += 1
                        passed = False
                        break
                    if _label_value(row, "quality") < min_assistant_quality:
                        rejected["assistant_quality"] += 1
                        passed = False
                        break
                    if _label_value(row, "toxicity") > max_toxicity:
                        rejected["assistant_toxicity"] += 1
                        passed = False
                        break
                reject_reason = _chat_compact_reject_reason(text)
                if reject_reason:
                    rejected[reject_reason] += 1
                    passed = False
                    break

            if not passed:
                continue

            block = _format_path(path, messages)
            if block in seen_blocks:
                rejected["duplicate"] += 1
                continue
            seen_blocks.add(block)
            out.write(block)
            out.write("\n")
            n_paths += 1
            n_chars += len(block) + 1
            length_hist[len(path)] += 1

    return {
        "messages": len(messages),
        "transcripts": n_paths,
        "chars": n_chars,
        "rejected": dict(sorted(rejected.items())),
        "length_hist": dict(sorted(length_hist.items())),
    }


def build_corpus(
    *,
    output: Path,
    lang: str,
    charset: str,
    all_paths: bool,
    max_chars: int | None,
    profile: str,
    window_messages: int,
    min_user_chars: int,
    max_user_chars: int,
    min_assistant_chars: int,
    max_assistant_chars: int,
    min_user_quality: float,
    min_assistant_quality: float,
    max_toxicity: float,
) -> dict[str, object]:
    source = hf_hub_download(
        repo_id="OpenAssistant/oasst1",
        repo_type="dataset",
        filename=READY_MESSAGES,
    )
    messages = _load_messages(source, lang=lang, charset=charset)
    if profile == "chat-compact":
        return _build_chat_compact_corpus(
            messages=messages,
            output=output,
            window_messages=window_messages,
            min_user_chars=min_user_chars,
            max_user_chars=max_user_chars,
            min_assistant_chars=min_assistant_chars,
            max_assistant_chars=max_assistant_chars,
            min_user_quality=min_user_quality,
            min_assistant_quality=min_assistant_quality,
            max_toxicity=max_toxicity,
        )
    children = _children_by_parent(messages)
    roots = _roots(messages)

    output.parent.mkdir(parents=True, exist_ok=True)
    n_paths = 0
    n_chars = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for root in roots:
            for path in _paths_from(root, children, [], all_paths=all_paths):
                # Single-message roots are prompts without assistant learning signal.
                roles = {messages[mid]["role"] for mid in path}
                if "prompter" not in roles or "assistant" not in roles:
                    continue
                block = _format_path(path, messages)
                if max_chars is not None and n_chars + len(block) > max_chars:
                    return {
                        "messages": len(messages),
                        "transcripts": n_paths,
                        "chars": n_chars,
                    }
                out.write(block)
                out.write("\n")
                n_paths += 1
                n_chars += len(block) + 1
    return {"messages": len(messages), "transcripts": n_paths, "chars": n_chars}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="training_data/oasst1_en_ready_chat.txt",
        help="Output .txt path for YAUM.",
    )
    parser.add_argument("--lang", default="en", help="OASST language code.")
    parser.add_argument(
        "--charset",
        choices=["ascii", "raw"],
        default="ascii",
        help="Character normalization for char-level YAUM training.",
    )
    parser.add_argument(
        "--best-paths-only",
        action="store_true",
        help="Follow only the first ranked child per conversation branch.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional character limit for fast trial corpora.",
    )
    parser.add_argument(
        "--profile",
        choices=["default", "chat-compact"],
        default="default",
        help="Corpus shaping preset. 'chat-compact' keeps short recent chat windows.",
    )
    parser.add_argument(
        "--window-messages",
        type=int,
        default=4,
        help="For chat-compact, keep at most this many recent messages per sample.",
    )
    parser.add_argument("--min-user-chars", type=int, default=8)
    parser.add_argument("--max-user-chars", type=int, default=280)
    parser.add_argument("--min-assistant-chars", type=int, default=48)
    parser.add_argument("--max-assistant-chars", type=int, default=900)
    parser.add_argument("--min-user-quality", type=float, default=0.2)
    parser.add_argument("--min-assistant-quality", type=float, default=0.45)
    parser.add_argument("--max-toxicity", type=float, default=0.5)
    args = parser.parse_args()

    stats = build_corpus(
        output=Path(args.output),
        lang=args.lang,
        charset=args.charset,
        all_paths=not args.best_paths_only,
        max_chars=args.max_chars,
        profile=args.profile,
        window_messages=args.window_messages,
        min_user_chars=args.min_user_chars,
        max_user_chars=args.max_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        max_assistant_chars=args.max_assistant_chars,
        min_user_quality=args.min_user_quality,
        min_assistant_quality=args.min_assistant_quality,
        max_toxicity=args.max_toxicity,
    )
    print(
        f"Wrote {args.output} | profile={args.profile} | messages={stats['messages']:,} "
        f"transcripts={stats['transcripts']:,} chars={stats['chars']:,}"
    )
    if "length_hist" in stats:
        print(f"Window lengths: {stats['length_hist']}")
    if "rejected" in stats:
        top_rejections = sorted(
            stats["rejected"].items(), key=lambda item: item[1], reverse=True
        )[:10]
        print(f"Top rejections: {top_rejections}")


if __name__ == "__main__":
    main()
