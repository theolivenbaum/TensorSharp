#!/usr/bin/env python3
"""What djev compiles for every public JevBench decision, with DiffusionGemma's real tokenizer.

For each task this drives djev's DiffusionEngine (djev/engine.py, unmodified) with the tokenizer djev pins
(google/diffusiongemma-26B-A4B-it at revision f7f5b7f5, transformers 5.17.0) and the token_ids transport djev
serves with, and records the compiled answer template, the slots and their label ids, the canvas width, the
seed-0 canvas, and the exact prompt ids djev would send - apply_chat_template of [system, user] with
enable_thinking=False. Prompts are stored as a SHA-256 of the id list (and in full, ids and text, for a few
tasks) to keep the file small. gemma_chat_template.jinja beside this file is that revision's chat_template.jinja,
so the text rendering can be checked without a checkpoint.

DjevJevBenchTests.GemmaCompilesExactlyWhatDjevCompiles loads a DiffusionGemma GGUF and must reproduce all of
it with TensorSharp's GGUF tokenizer and chat template.

    pip install "transformers==5.17.0" pydantic httpx
    # tokenizer.json, tokenizer_config.json, chat_template.jinja from the pinned revision in <tokenizer-dir>
    python3 generate_gemma_fixtures.py <djev-dev> <jevbench> <tokenizer-dir> > gemma_fixtures.json
"""
import hashlib
import json
import sys
from pathlib import Path

djev_dir, jevbench_dir, tokenizer_dir = sys.argv[1:4]
sys.path.insert(0, djev_dir)

from transformers import AutoTokenizer  # noqa: E402
from djev.contracts import DjevRequest  # noqa: E402
from djev.engine import DiffusionEngine, _messages  # noqa: E402

FULL_PROMPTS = 2


def main():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    engine = DiffusionEngine(tokenizer=tokenizer, canvas=128, compact=True, max_model_len=32768,
                             input_transport="token_ids")
    out = []
    # Prompts kept in full: the first few, plus the first of each shape worth seeing rendered.
    kinds_seen = set()
    for split in ("easy", "hard", "original"):
        for line in Path(jevbench_dir, "datasets", "public", f"{split}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            question = {"type": task["question"]["type"], "instructions": task["question"]["instructions"]}
            if task["question"].get("criteria") is not None:
                question["criteria"] = task["question"]["criteria"]
            request = DjevRequest.model_validate({"state": task["state"], "questions": {"decision": question}})
            compiled = engine.compile(request)
            prepared = engine._check_context(compiled, engine._request_state(request))
            prompt = list(prepared.prompt_token_ids)
            record = {
                "id": task["id"],
                "template": list(compiled.template),
                "slots": [{"position": s.position, "token_ids": list(s.token_ids)} for s in compiled.slots],
                "width": compiled.canvas_width,
                "canvas_seed0": engine._canvas(compiled, 0),
                "prompt_tokens": len(prompt),
                "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            }
            kind = ("dict" if isinstance(task["state"], dict) else "list" if isinstance(task["state"], list) else "text",
                    question["type"], split)
            if len(out) < FULL_PROMPTS or kind not in kinds_seen:
                kinds_seen.add(kind)
                record["request"] = {"state": task["state"], "questions": {"decision": question}}
                record["prompt"] = prompt
                record["prompt_text"] = tokenizer.apply_chat_template(
                    _messages(compiled, engine._request_state(request)), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            out.append(record)
    json.dump({"generator": "generate_gemma_fixtures.py", "tokenizer": "google/diffusiongemma-26B-A4B-it@f7f5b7f5",
               "tasks": out}, sys.stdout, indent=0)


if __name__ == "__main__":
    main()
