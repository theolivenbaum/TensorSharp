#!/usr/bin/env python3
"""Golden fixtures for DjevParityTests, produced by djev's own engine.

Runs djev's DiffusionEngine (djev/engine.py, unmodified) against a mock vLLM transport and records, for each
case, every read it made - the system prompt, the seeded canvas, its width and the requested label ids - and
the response body it built from the mock's log probabilities. The C# tests replay the same cases through
TensorSharp.Structured.Decisions.DiffusionAgent with the same tokenizer and the same mock scores and must
reproduce all of it: prompts and canvases exactly, probabilities to 1e-12.

The tokenizer and the score function are deliberately simple and duplicated in the C# test: word pieces with
FNV-1a ids, and log probabilities that are multiples of 1/512 so they survive the float round trip exactly.

    pip install pydantic httpx
    python3 generate_djev_fixtures.py /path/to/djev-dev > djev_fixtures.json
"""
import asyncio
import json
import math
import re
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "../djev-dev")

import httpx  # noqa: E402
from djev.contracts import DjevRequest  # noqa: E402
from djev.engine import DiffusionEngine  # noqa: E402

VOCAB = 262144
PIECES = re.compile(r"<\|channel>|<channel\|>|[A-Za-z]+|\d|\s|.", re.DOTALL)


def piece_id(piece):
    h = 2166136261
    for b in piece.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return 300 + h % (VOCAB - 300)


class WordTokenizer:
    vocab_size = VOCAB

    def encode(self, text, add_special_tokens=False):
        return [piece_id(m.group(0)) for m in PIECES.finditer(text)]


def score(canvas, position, token_id):
    h = 1469598103934665603
    for value in [*canvas, position, token_id]:
        h = ((h ^ (value & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 29
    return -((h % 4096) / 512.0) - 0.125


def response(body, reads):
    xargs = body["vllm_xargs"]
    canvas = xargs["diffusion_seed_canvas"]
    system = body["messages"][0]["content"][0]["text"]
    user = body["messages"][1]["content"][0]["text"]
    reads.append({"system": system, "user": user, "canvas": canvas, "width": xargs["diffusion_canvas_length"],
                  "label_ids": body["logprob_token_ids"], "max_tokens": body["max_tokens"]})
    rows = []
    for position in range(len(canvas)):
        entries = [{"token": f"token_id:{t}", "logprob": score(canvas, position, t)} for t in body["logprob_token_ids"]]
        rows.append({"token": f"token_id:{canvas[position]}", "logprob": score(canvas, position, canvas[position]),
                     "top_logprobs": entries})
    return {"choices": [{"index": 0, "logprobs": {"content": rows}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(canvas), "total_tokens": 20 + len(canvas)}}


TRIAGE = {
    "intent": {"type": "choice", "instructions": "What does the customer want in `message`?",
               "criteria": {"refund": "money returned or a duplicate charge reversed",
                            "technical_help": "a bug, outage or integration problem",
                            "other": None}},
    "is_urgent": {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"},
    "frustration": {"type": "score", "instructions": "How frustrated does the customer sound?",
                    "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"]},
}

CASES = [
    {"name": "joint-default", "compact": True,
     "request": {"state": "I was charged twice and nobody answers", "questions": TRIAGE}},
    {"name": "joint-samples-seed", "compact": True,
     "request": {"state": "I was charged twice and nobody answers", "questions": TRIAGE,
                 "options": {"samples": 3, "seed": 17, "diagnostics": True}}},
    {"name": "joint-negative-seed-not-compact", "compact": False,
     "request": {"state": "Checkout is down.", "questions": TRIAGE, "options": {"seed": -3, "diagnostics": True}}},
    {"name": "joint-big-seed", "compact": True,
     "request": {"state": "Checkout is down.", "questions": {"urgent": TRIAGE["is_urgent"]},
                 "options": {"seed": 123456789012345678901234567890}}},
    {"name": "structured-state-and-criteria", "compact": True,
     "request": {
         "state": {"message": "Café «refund» — ticket #4411", "amount": 12.5, "items": [1, 2.0, 1e-05, 1e16],
                   "nested": {"flag": True, "none": None, "quote": "say \"hi\"\n\ttab"}},
         "questions": {
             "decision": {"type": "choice",
                          "instructions": {"task": "route the ticket", "rules": ["billing first", "then support"]},
                          "criteria": {"billing": {"covers": ["refunds", "charges"]}, "support": "",
                                       "security": ["phishing", 3]}},
             "policy_ok": {"type": "noul", "instructions": "Is the refund permitted?",
                           "criteria": {"true": "Every condition holds.", "false": None}},
             "single": {"type": "choice", "instructions": "Only one option", "criteria": {"only": "the one"}},
         },
         "options": {"diagnostics": True}}},
    {"name": "independent-duplicates", "compact": True,
     "request": {"state": "I was charged twice and nobody answers",
                 "questions": {**TRIAGE, "urgent_again": TRIAGE["is_urgent"]},
                 "options": {"isolation": "independent", "samples": 2, "diagnostics": True}}},
    {"name": "independent-score-levels", "compact": True,
     "request": {"state": "The roster shows two rest violations.",
                 "questions": {"violations": {"type": "score", "instructions": "How many violations?",
                                              "criteria": ["none", "one", "two", "three or more"]},
                               "urgent": TRIAGE["is_urgent"]},
                 "options": {"isolation": "independent", "score_mode": "independent_levels", "samples": 2,
                             "seed": 5, "diagnostics": True}}},
    {"name": "many-options", "compact": True,
     "request": {"state": "pick", "questions": {
         "wide": {"type": "choice", "instructions": "Pick one", "criteria": {f"opt{i}": None for i in range(30)}}}}},
]


async def run(case):
    reads = []

    async def handler(request):
        return httpx.Response(200, json=response(json.loads(request.content), reads))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = DiffusionEngine(tokenizer=WordTokenizer(), client=client, canvas=128, compact=case["compact"],
                                 max_model_len=32768, max_request_reads=1)
        result = await engine.generate(DjevRequest.model_validate(case["request"]))
    body = result.body
    body.pop("usage", None)
    return {**case, "reads": reads, "response": body}


def main():
    out = [asyncio.run(run(case)) for case in CASES]
    json.dump({"generator": "generate_djev_fixtures.py", "cases": out}, sys.stdout, ensure_ascii=False, indent=1,
              allow_nan=False)


if __name__ == "__main__":
    main()
