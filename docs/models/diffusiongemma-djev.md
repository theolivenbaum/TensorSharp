# DiffusionGemma typed decisions: alignment with djev

[← back to DiffusionGemma](diffusiongemma.md)

[djev](https://github.com/Davipar/djev-dev) is the reference implementation of typed decisions on
DiffusionGemma: Noul / Choice / Score questions compiled into a one-step structured read of exact label
probabilities, served through a patched vLLM. This page reviews what TensorSharp shipped before
(`TensorSharp.Structured`'s JSON canvas, commit `b70d745`) against djev, and describes the djev-aligned
path that now sits beside it in `TensorSharp.Structured.Decisions`.

## Summary

| | JSON canvas (`StructuredPredictor`) | djev, and `DiffusionAgent` |
|---|---|---|
| Lineage | open-jev | djev (`djev/engine.py`, `djev/contracts.py`) |
| Prompt | one user turn: `<user_text>` + `<instructions>` blocks | system turn: preamble, `Question i:` with labelled criteria, reply format; user turn: the state |
| Answer canvas | every allowed answer tokenized as a whole JSON document; agreeing positions pinned | `<\|channel>thought\n<channel\|>` + one `i:label` line per question, then `<turn\|>` (106) and `<pad>` (0) |
| Labels | the option values themselves, often several tokens | one token each: `no`/`yes`, `A`..`Z`,`AA`..`JB`, `0`..`9`; option names stay in the prompt |
| Canvas width | the served 256, or a tight fit | template + 1 rounded up to a multiple of 16, at most 128 |
| Slot noise | the sampler's RNG | CPython `random.Random("djev-canvas-v1:{seed}").randrange(262144)` |
| Readout | greedy walk over free positions; probability = softmax at the first discriminating position | softmax over the exact label ids at the one slot, normalised over the allowed labels |
| Output | `{"key": value}` JSON plus per-option scores | djev's typed answers: `noul` = P(yes); `choice` + probabilities + confidence; `score` = expected level + legend + probabilities + confidence |
| Samples | – | 1–4 reads averaged, seeds `seed + i·7919` |
| Isolation | questions that do not fit are split across canvases | joint (one canvas, error if it does not fit) or independent (one read per distinct question, SHA-256-derived seeds) |
| Score mode | categorical | categorical, or `independent_levels` (`truth-odds-v1`) |
| Limits and errors | minimal | djev's: 32 questions, 255 options, 2–10 levels, 20 000 / 2 000 / 500 characters; schema errors (422) before any read; missing or invalid evidence (502) never papered over |

The low-level read in `TensorSharp.Models` was already the right primitive: `DiffusionReadOptions` with
`ReadOnly`, `MaxSteps = 1`, a full `SeedCanvas`, a per-request `CanvasWidth` and per-position
`LogprobTokenIds` is djev's `diffusion_read_only` / `diffusion_max_steps` / `diffusion_seed_canvas` /
`diffusion_canvas_length` / `logprob_token_ids`, reported at temperature 1. What differed was everything
built on top of it.

## Findings

1. **Different construction.** The JSON canvas is open-jev's idea, not djev's. It asks the model to write
   JSON, pins the scaffolding and walks multi-token values; djev asks for one label token per question and
   reads its distribution directly. They produce different prompts, different canvases and different
   numbers, so a receipt from one says nothing about the other. The JSON canvas is left as it was, and
   documented as the open-jev construction; the djev path is new.
2. **Prompt ended on the wrong token.** `ChatTemplate.RenderFromGgufTemplate` kept the template's trailing
   `\n` after `<|turn>model` only for architecture `gemma4` and `TrimEnd()`ed every other family - so every
   DiffusionGemma prompt, chat included, ended `<|turn>model` where the published template (and vLLM) end
   `<|turn>model\n`. Fixed for `diffusion-gemma`, which the rest of the code already treats as Gemma 4's
   template family.
3. **String content is not content parts.** djev sends vLLM content parts, and Gemma 4's template spells a
   system *text part* as `trim(text) + ' '` but a system *string* as `trim(text)`. TensorSharp's messages
   are strings, so its render lacked the space djev's prompts carry before `<turn|>`.
   `DecisionPrompt.Render` respells it.
4. **Silent truncation.** `ChatGenerationPipeline.DiffusionReadAsync` passes the rendered prompt through
   `TruncatePromptToContext`, so an over-long read is answered about a shortened prompt. djev refuses
   instead. The decision path uses a new `DiffusionReadTokensAsync`, which takes the tokenized prompt as is,
   and the agent refuses a prompt plus canvas beyond the context before any read.
5. **No evidence checks.** djev fails a read that omits a requested label, reports NaN, +∞ or a positive
   log-probability, or has no finite label at all, and treats vLLM's −9999 as impossible rather than as
   evidence. The JSON readout took whatever the scores were. `DiffusionAgent` applies djev's checks.
6. **Images.** djev reads state, question and option images natively. TensorSharp's DiffusionGemma is
   text-only, so an image anywhere in a request is refused (422), as djev's text-only base engine does.

## What is reproduced, and how it is checked

Everything that is a function of the request, the tokenizer and the logits is reproduced exactly:

- **djev's own engine as the oracle.** `InferenceWeb.Tests/Fixtures/Djev/generate_djev_fixtures.py` runs
  `djev/engine.py` unmodified against a mock vLLM transport and records every read (system prompt, user
  turn, seed canvas, width, label ids, `max_tokens`) and the response it built. `DjevDecisionTests`
  replays eight cases - joint, three samples, a negative and a 30-digit seed, non-compact, structured
  state and criteria with floats and non-ASCII, independent isolation with a duplicate question,
  `independent_levels`, and 30 options - and must match prompts and canvases exactly and every
  probability and diagnostic to 1e-12.
- **The real tokenizer.** `generate_gemma_fixtures.py` runs djev's compiler with the tokenizer djev pins
  (`google/diffusiongemma-26B-A4B-it@f7f5b7f5`, transformers 5.17.0) over all 231 public JevBench
  decisions. `DjevGemmaParityTests` checks the rendered prompt text through TensorSharp's Jinja engine and
  the pinned `chat_template.jinja` without a checkpoint, and - with `TS_TEST_MODEL_DIR` - the answer
  templates, slots, seed canvases and prompt token ids against a GGUF's tokenizer.
- **Bit-exact pieces.** CPython's MT19937 seeding and `randrange`, `json.dumps(ensure_ascii=False,
  separators=(",", ":"))` including float `repr`, `math.fsum`, the SHA-256 seed derivation.

The constants djev hard-codes hold for this vocabulary: 106 is `<turn|>`, 0 is `<pad>`, the vocabulary is
262 144, and every label in djev's list is one token in context.

## What still differs

- **Arithmetic.** djev serves BF16 weights and a BF16 KV cache on vLLM; a GGUF here is usually quantized
  (Q4_K_M is the smallest), on different kernels. Same prompt, same canvas, different logits: expect the
  probabilities to move and some answers to flip. Neither side promises bitwise repeatability.
- **Batching.** djev bounds concurrent reads and relies on vLLM's scheduler; here a batch of reads is one
  denoising block (`DiffusionAgentOptions.BatchSize`, or the server's diffusion scheduler). The arithmetic
  of a canvas can depend on what it was batched with, on both sides.
- **Prefix caching.** djev reuses eligible KV prefixes across requests. TensorSharp prefills each read's
  prompt; answers are unaffected, latency is.
- **Images**, as above.

## Using it

```csharp
using TensorSharp.Structured.Decisions;

using var agent = DiffusionAgent.Load("models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf", BackendType.GgmlCuda);
var result = agent.SystemOne("I was charged twice and nobody answers", Presets.Triage());
result["intent"].Choice;          // the winning option
result["refund_requested"].Noul;  // P(yes), relative to no/yes
result.ToJsonString();            // djev's response body
```

Over HTTP, `TensorSharp.Server` answers djev's `POST /v1/request` with djev's request and response bodies,
so a client written for djev - jevbench's djev adapter included - can point at it. `GET /v1/request/config`
reports the limits.

The JevBench harness is [`benchmarks/DiffusionDecisionBench`](../../benchmarks/DiffusionDecisionBench/README.md).
