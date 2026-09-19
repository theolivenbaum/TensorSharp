# DiffusionGemma

[← back to model index](README.md)

## Status snapshot

| Field | Status |
|---|---|
| GGUF architecture keys | `diffusion-gemma`, `diffusion_gemma` |
| Source class | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| Sampler | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| Modalities | Text only |
| Thinking / tools | Thought channel parsed out (returned only on `"think": true`); tools/tool_choice refused with HTTP 400 |
| Generation mode | Block text diffusion, not autoregressive token decode |
| CLI support | `TensorSharp.Cli` detects `DiffusionGemmaModel` and uses diffusion run mode |
| Server support | Web UI chat stream with live denoising previews; Ollama/OpenAI compatibility endpoints use append-oriented response shapes and return the final text only (no denoising previews) |
| Continuous batching | Dedicated [`DiffusionBatchScheduler`](../../TensorSharp.Chat/DiffusionBatchScheduler.cs), admitted at block boundaries |

## Downloads

Verified GGUF pointers:

| Model | HF repo | Recommended file | Notes |
|---|---|---|---|
| diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16.807 GB); also `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16` | GGUF `general.architecture` = `diffusion-gemma`. Official upstream weights: [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |

`Q4_K_M` is the smallest published quant. No companion files are needed
(text only — no mmproj).

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
```

CLI diffusion mode (auto-dispatched from the model's architecture — no mode
flag; the prompt comes from a file via `--input`):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt \
  --backend ggml_cuda --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

Server (the Web UI at `http://localhost:5000/index.html` streams live denoising previews —
each step repaints the whole message via `replace` SSE frames; the Ollama/OpenAI
compatibility endpoints return the final text only):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

## 1. Origin and intent

DiffusionGemma is a block text-diffusion language model built on a Gemma-4-style
Mixture-of-Experts backbone. It is not the same runtime contract as the
autoregressive `gemma4` model:

- `Forward(int[] tokens)` intentionally throws. Generation must go through
  `DiffusionGemmaSampler`.
- Each denoising step runs over a concatenated `[prompt | canvas]` sequence.
- The prompt side is causal and never attends to the canvas.
- The canvas side is bidirectional over the prompt and canvas.
- The emitted block is the current deterministic argmax canvas, refined over
  multiple denoising steps.

The GGUF file must report `general.architecture=diffusion-gemma` or
`diffusion_gemma`; `ModelBase.Create()` routes those keys to
`DiffusionGemmaModel`.

## 2. Forward graph

The model exposes two execution regimes.

The unified correctness path is `ForwardCanvas(tokens, promptLen)`:

```text
[prompt tokens | canvas tokens]
  -> region-aware embedding scale
  -> prompt/canvas attention masks
  -> N Gemma-style transformer layers
       - local/global QK-norm attention
       - dense gated-GELU MLP
       - top-k MoE experts
       - prompt encoder scale / canvas decoder scale
  -> output norm
  -> tied lm-head
  -> final logit softcap
  -> canvas logits
```

The optimized GPU path splits each block into a prompt prefill plus repeated
canvas decodes:

1. `PrefillPrompt(promptTokens)` computes the prompt K/V once.
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` reuses prompt K/V
   for every denoising step.
3. The sampler accepts low-entropy positions, re-noises the rest, and repeats.

Prompt-KV caching is enabled on device-glue backends and bypassed on the pure
CPU path.

## 3. Sampler contract

`DiffusionEbParams` controls generation:

| Parameter | Default | Meaning |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | Maximum refinement steps per canvas block |
| `TMin` / `TMax` | 0.4 / 0.8 | Temperature schedule from late to early denoising |
| `EntropyBound` | 0.1 | Cumulative mutual-information bound for accepted positions |
| `StabilityThreshold` | 1 | How many stable argmax steps are required before early stop |
| `ConfidenceThreshold` | 0.005 | Mean entropy threshold for early stop |
| `Seed` | 0 | Deterministic sampler seed |
| `MaxBlocks` | 1 | Number of block-autoregressive canvas blocks |

The CLI maps this through:

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

When `--diffusion-blocks` is `0`, the CLI derives the number of blocks from
`--max-tokens` and `diffusion.canvas_length`.

## 4. Architecture details

DiffusionGemma reuses many Gemma-4 backbone choices:

- NeoX RoPE with separate local/global dimensions.
- Five local sliding-window layers followed by one global layer pattern.
- Per-head Q/K RMSNorm and unweighted V RMSNorm.
- Global layers can omit `attn_v.weight`, using raw K as V.
- Dense gated-GELU MLP plus 128-expert top-8 MoE.
- Tied embeddings / lm-head and final logit softcapping.

Diffusion-specific metadata includes:

| Key | Meaning |
|---|---|
| `diffusion.canvas_length` | Number of canvas positions denoised per block, default 256 |
| `tokenizer.ggml.mask_token_id` | Mask token id used by warmup and fallback paths |
| `<arch>.attention.sliding_window_pattern` | Local/global layer pattern |
| `<arch>.attention.head_count_kv` | Per-layer KV head counts |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert count and active top-k |

## 5. Acceleration status

Current optimized paths include:

- Prompt-KV cache for GPU backends.
- Self-conditioning enabled by default; disable with `DIFFUSION_NO_SC=1`.
- GGML fused decode layer, fused whole-model decode, and fused lm-head tail.
- CUDA VRAM residency planning: when the model is larger than VRAM, weights are
  preloaded device-side in priority order (lm_head/embedding, per-layer
  attention/dense, then MoE expert stacks) up to free-VRAM-minus-headroom, the
  device-copy cache is capped, and decode switches to the SEGMENTED per-layer
  fused path so the non-resident remainder streams through one bounded staging
  buffer instead of oversubscribing VRAM (which makes Windows WDDM page the
  working set every submission — measured ~4x slower than streaming).
- Step-invariant decode masks are cached host-side and bound cacheable (one
  device upload per block geometry instead of a rebuild+upload per layer/step).
- SIMD-vectorized host paths (`TensorPrimitives`): per-position
  argmax/entropy/multinomial sampling and the final-logit softcap; the fused
  lm-head logits land in one pooled pinned buffer instead of a fresh 268 MB
  allocation per step.
- MLX K-quant affine repacking for DiffusionGemma's multi-row canvas workload.
- Block-boundary continuous batching in `TensorSharp.Server` through
  `DiffusionBatchScheduler`.

Important toggles:

| Variable | Effect |
|---|---|
| `DIFFUSION_STEPS` | Server-side denoising steps per block, default 48 |
| `DIFFUSION_MAX_BATCH` | Server diffusion scheduler max active requests, default 2 |
| `DIFFUSION_NO_PKV=1` | Disable prompt-KV caching on device-glue backends |
| `DIFFUSION_NO_SC=1` | Disable self-conditioning |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff, default 32 |
| `DIFFUSION_BATCHED_FORWARD=1` | Use true batched canvas decode instead of time-sliced fused single-canvas decode |
| `DIFFUSION_NO_FUSED_DECODE=1` | Disable GGML fused whole-model diffusion decode |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL=1` | Disable fused output-norm + lm-head + softcap tail |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Cap transient batched lm-head logits memory, default 300 MB |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda: VRAM kept free of preloaded weights, default 2048 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda: device-copy cache cap when the model spills VRAM, default 768 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda: force per-layer fused decode `1`/`0` (auto when the model spills VRAM) |
| `DIFFUSION_PIN_STREAMED=1` | ggml_cuda: page-locked copies of streamed weights for DMA uploads (costs RAM) |

## 6. Server behavior

When the Web UI hosts a DiffusionGemma GGUF:

- `/api/chat` takes the diffusion path.
- The stream emits `replace` events rather than token append events, because
  every denoising step refines the whole current canvas.
- A final replacement is emitted before the `done` event.
- Concurrent requests share one background diffusion scheduler and are admitted
  between blocks.
- On backends without prompt-KV caching (`cpu`, `ggml_cpu`) the scheduler runs
  each sequence's step through the unified `[prefix|canvas]` forward instead of
  prefill + canvas decode; behavior and output are identical.

The Ollama and OpenAI compatibility adapters still use append-oriented response
shapes through `ChatStreamWithMetricsAsync`. They can surface the final
DiffusionGemma text, but the live denoising previews and `replace` frames are
Web UI-only.

The model writes Gemma 4's channel syntax: a canvas may open with the
`<|channel>thought\n` primer, or close a thought block the prompt opened with a
bare `<channel|>`, before the answer. The architecture is registered as the
`diffusion-gemma` chat protocol (`Gemma4OutputParser`, always required, the
GGUF template still renders the prompt), and every preview and the final text
go through that parser: the thought block is dropped unless the request asks for
reasoning (`"think": true` returns it as `reasoning_content`), and the channel
markers never reach a client. Before this the raw canvas was delivered verbatim
and OpenAI answers began with the literal `<|channel>thought` marker.

Tool calling is refused up front: `/v1/chat/completions` answers HTTP 400
(`{"error": ...}`, `invalid_request_error`) to any request that carries `tools`
or a `tool_choice` other than `"none"` while a DiffusionGemma model is loaded,
because a block-diffusion turn has no tool loop to feed a result back into.
`/v1/responses` and Ollama's `/api/chat` refuse `tools` the same way. The
built-in skills / code-execution tools are never offered to this family either
(the protocol entry declares `RendersToolDeclarations = false`), so `--code-exec`
and skills discovery leave a diffusion request exactly as it was before.

## 6a. Structured reads

A discrete diffusion model denoises a whole canvas per forward pass. Seed the
canvas with the answer's fixed text, leave only the slots you want read as
noise, and one denoise step yields a distribution over each of those slots -
a classifier, not a continuation. TensorSharp exposes that as a **read**.

Three request fields carry it, named after the `extra_args` that vLLM accepts
for the same model, so a schema layer written against either engine drives the
other unchanged:

| field | type | meaning |
|---|---|---|
| `diffusion_seed_canvas` | `int[]`, exactly the canvas width | replaces the random initial canvas after prefill |
| `diffusion_canvas_length` | `int` | the leading canvas positions this request owns (default: the served canvas) |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas at the cap, end the request there, and report temperature-1 logprobs at every position |

A read differs from a generation in three ways:

- **It ends on the canvas it emits.** `MaxBlocks` is pinned to 1, and the
  canvas is *not* trimmed at an end-of-turn token: the canvas is the answer, so
  a stray end token drawn into a noise slot must not cut it short.
- **It reports at temperature 1**, not at the step's place on the denoising
  schedule. The schedule exists to make sampling converge; tempering the
  reported numbers would rescale the very probabilities the caller asked for.
  The argmax is the same either way, so the emitted canvas and the reported
  distribution always agree on the most likely token.
- **Its step cap is its own.** A read capped at one step leaves the batch after
  that step even when a generation alongside it keeps denoising.

Because only the full logits carry a distribution, a read that asks for
logprobs stays on the host logits path rather than the on-device sampler (whose
top-K is taken at the step's temperature). That is one readback per step, over
the handful of steps a read runs. The top-K sweep itself runs once per read,
on the step that emits.

Note one difference from vLLM: TensorSharp's canvas forward is fixed-width, so
`diffusion_canvas_length` narrows what the request *owns* rather than what is
computed. Positions past the width are re-noised every step - they never settle
and are never emitted - but they cost the same as a full canvas.

### In-process

```csharp
var options = new DiffusionReadOptions
{
    ReadOnly = true,
    MaxSteps = 1,
    CanvasWidth = 16,
    SeedCanvas = seedIds,   // exactly 16 ids
    TopLogprobs = 20,
};
DiffusionReadResult read = await modelService.DiffusionReadAsync(
    session, history, options, seed: 0, cancellationToken);
```

`DiffusionReadResult` carries the emitted `Canvas`, one
`DiffusionPositionLogprobs` per position (token ids and natural log
probabilities, most likely first), the steps actually run and whether the
canvas converged rather than hitting its cap. `DiffusionGemmaSampler.Read`
is the same thing one level down, against raw prompt tokens.

### Over HTTP

`POST /v1/diffusion/read`. Chat completions cannot carry this shape - there is
no continuation and no finish reason, only per-position logprobs - so it has
its own route.

```bash
curl -s localhost:8080/v1/diffusion/read -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\"}]}"},
    {"role": "user", "content": "Everything is down and we have a demo at noon."}
  ],
  "diffusion_seed_canvas_text": "The answer is",
  "diffusion_max_steps": 1,
  "top_logprobs": 20,
  "seed": 0
}'
```

The seed canvas goes in as `diffusion_seed_canvas` (token ids) or
`diffusion_seed_canvas_text` (tokenized server-side, `addSpecial: false`); give
one, not both. A malformed read is a `400` with an `invalid_request_error`
before anything reaches the sampler - an id outside the vocabulary is an
out-of-bounds embedding lookup on the device, and a canvas whose length does not
match its width would read back different slots than the caller wrote. The
response is a `diffusion.read` object: the canvas as tokens and decoded text,
`steps`, `converged`, and a `logprobs` array of `{position, top_logprobs}`.

## 6b. Typed JSON decisions

A read whose canvas is a JSON *template* turns the model into a classifier.
Tokenize every allowed answer as a complete JSON document, pin the token
positions all of them agree on - the braces, the quoted keys, the separators -
and leave free only the positions where they differ. One denoise step later,
the scores at those free positions choose among the allowed tokens, and the
answer is a complete member of the allowed language by construction: no JSON
repair, no retry, no second pass.

[`TensorSharp.Structured`](../../TensorSharp.Structured/README.md) is that
layer, with a benchmark harness for accuracy, throughput and cost. Its
construction follows [open-jev](https://github.com/theolivenbaum/open-jev), the
Python research harness for the same idea on this model, and its benchmark
receipts use open-jev's field names so runs can be put side by side.

```csharp
var predictor = new StructuredPredictor(new DiffusionGemmaReader(model));
StructuredPrediction answer = await predictor.PredictAsync(new StructuredRequest
{
    Document = "I was charged twice. Please refund the duplicate.",
    Questions = new Dictionary<string, StructuredQuestion>
    {
        ["refund_requested"] = StructuredQuestion.Boolean("Does the customer ask for a refund?"),
        ["department"] = StructuredQuestion.Choice("Which team?", "billing", "technical", "sales"),
    },
});
```

Two read fields exist for it, on top of the structured-read contract above:

| field | meaning |
|---|---|
| `PinnedPositions` | canvas positions held at their seed value for the whole denoise. A pinned position contributes no entropy and settles immediately - the cheap form of a logits mask allowing exactly one token there. Free positions denoise unrestricted. |
| `LogprobTokenIds` | per position, the token ids to report the score of, instead of that position's top-K. A constrained readout needs the score of every *allowed* token at a slot, and an allowed token can sit far outside any top-K. |

Where open-jev applies a `[canvas, vocab]` logits mask each step, TensorSharp
pins positions: the same effect where it matters, without materializing the
mask. Questions on one canvas share attention, and a question set too large for
one canvas is split and merged - so this does not isolate questions from one
another.

## 7. Test coverage

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) is
opt-in on real GGUFs via `TS_TEST_MODEL_DIR`. It covers:

- `ForwardCanvas` finite-logit correctness.
- End-to-end EntropyBound generation.
- Prompt-KV equivalence and speed probes.
- Regression guards for repeated-token output and device-memory retention.
- Batched decode equivalence and two-request generation through the scheduler
  style used by the server.
- Structured reads: the emitted canvas and its per-position distribution, the
  read's equivalence between the single-request and batched paths, and the
  per-request step cap in a batch with a longer generation.
- Typed JSON decisions: pinned positions holding through a multi-step denoise,
  a prediction in its allowed language, and a benchmark receipt.

[`DiffusionStructuredReadTests`](../../InferenceWeb.Tests/DiffusionStructuredReadTests.cs)
needs no checkpoint and runs in ordinary CI: what a read is allowed to ask for
(and the refusals), what it does to the sampler parameters, and the temperature-1
top-K itself.

[`StructuredDecisionTests`](../../InferenceWeb.Tests/StructuredDecisionTests.cs)
also runs without a checkpoint: a character tokenizer compiles the canvases for
real, and a scripted reader stands in for the denoise, so the canvas, the
constrained readout, the canvas packing and the benchmark harness are exercised
on their own terms.

## 8. Remaining work

- Add dedicated API examples once Ollama/OpenAI adapters grow a diffusion-aware
  compatibility surface.
- Let `diffusion_canvas_length` narrow the canvas forward itself, not only what
  the request owns, so a short read stops paying for the served canvas.
- Promote true batched canvas decode only if it wins on target GPUs; today the
  fused single-canvas path can be faster when one canvas already saturates the
  GPU.
- Fold more diffusion scheduler metrics into `/api/queue/status` if operators
  need per-diffusion-batch visibility.
