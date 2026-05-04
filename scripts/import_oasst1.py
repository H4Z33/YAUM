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
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


READY_MESSAGES = "2023-04-12_oasst_ready.messages.jsonl.gz"
ROLE_LABELS = {
    "prompter": "<|user|>",
    "assistant": "<|assistant|>",
}


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _load_messages(path: str, lang: str) -> dict[str, dict]:
    messages: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("deleted"):
                continue
            if row.get("lang") != lang:
                continue
            role = row.get("role")
            text = _clean_text(row.get("text") or "")
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


def build_corpus(
    *,
    output: Path,
    lang: str,
    all_paths: bool,
    max_chars: int | None,
) -> dict[str, int]:
    source = hf_hub_download(
        repo_id="OpenAssistant/oasst1",
        repo_type="dataset",
        filename=READY_MESSAGES,
    )
    messages = _load_messages(source, lang=lang)
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
    args = parser.parse_args()

    stats = build_corpus(
        output=Path(args.output),
        lang=args.lang,
        all_paths=not args.best_paths_only,
        max_chars=args.max_chars,
    )
    print(
        f"Wrote {args.output} | messages={stats['messages']:,} "
        f"transcripts={stats['transcripts']:,} chars={stats['chars']:,}"
    )


if __name__ == "__main__":
    main()
