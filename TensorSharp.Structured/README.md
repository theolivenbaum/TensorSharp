# TensorSharp.Structured

Typed decisions on DiffusionGemma, two ways:

- **`TensorSharp.Structured.Decisions` - djev's structured read.** Noul / Choice / Score questions compiled
  into a one-token-per-question answer template, one denoising step, and the exact probabilities of each
  question's allowed label tokens. This follows [djev](https://github.com/Davipar/djev-dev), the reference
  implementation, prompt for prompt and canvas for canvas, and is the recommended path. It answers with
  djev's response shape, and `TensorSharp.Server` serves djev's `POST /v1/request`.
- **`StructuredPredictor` - the JSON canvas.** open-jev's construction: every allowed answer tokenized as a
  JSON document, the scaffolding pinned, a greedy readout over the free positions. Kept as it was; see
  [below](#the-json-canvas-open-jev).

How the two differ, and how the djev path is checked against djev's own engine, is in
[docs/models/diffusiongemma-djev.md](../docs/models/diffusiongemma-djev.md).

## Decisions (djev)

The API mirrors [Laya](https://github.com/theolivenbaum/laya)'s: a `QuestionSet`, one call, typed answers.

```csharp
using TensorSharp.Structured.Decisions;

using var agent = DiffusionAgent.Load("models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf", BackendType.GgmlCuda);

var result = agent.SystemOne("I was charged twice and nobody answers", Presets.Triage());

// (the values below are illustrative, not a measured output)
result["intent"].Choice;            // "refund"   - the most probable option
result["intent"].Probabilities;     // every option, in option order
result["frustration"].Score;        // 2.1        - the expected level on the 0..3 rubric
result["refund_requested"].Noul;    // 0.93       - P(yes), relative to no/yes
result["intent"].Confidence;        // 0.61       - entropy concentration, not calibration
result.Usage;                       // input tokens read, output tokens (the canvas template)
```

Questions are built the same way as in Laya:

```csharp
var questions = new QuestionSet()
    .Add("intent", Question.Choice("What does the customer want?",
        ("refund", "money returned or a duplicate charge reversed"),
        ("technical_help", "a bug, outage or integration problem"),
        ("other", null)))
    .Add("urgency", Question.Score("How urgent is this?",
        "no time pressure", "needs attention soon", "blocking issue or hard deadline"))
    .Add("permitted", Question.Noul("Is the refund permitted under the policy?",
        trueCriterion: "every condition holds", falseCriterion: "a condition is missing"));

// State is a string, or anything that serializes to a JSON object or array.
var answer = agent.SystemOne(new { subject = "Invoice #4411", body = "Billed twice." }, questions,
    new DecisionOptions { Samples = 2, Isolation = DecisionIsolation.Independent, Diagnostics = true });
```

`Presets` carries Laya's five sets (`Triage`, `Email`, `Guard`, `Moderation`, `ModelRouter`), so the same
workflow can be asked of both engines and compared.

| option | djev field | meaning |
|---|---|---|
| `Samples` | `samples` | 1-4 one-step reads, averaged |
| `Seed` | `seed` | the canvas seed (any integer; `null` draws one) |
| `Isolation` | `isolation` | `Joint`: one canvas; `Independent`: one read per distinct question |
| `ScoreMode` | `score_mode` | `IndependentLevels`: each level its own yes/no read, combined by odds (needs `Independent`) |
| `Diagnostics` | `diagnostics` | label mass, entropy, read counts, canvas width |

A djev request body works as is - `DecisionRequest.FromJson(body)` - and `result.ToJsonString()` is djev's
response body. Several requests batch into shared denoising blocks with
`agent.PredictAsync(IReadOnlyList<DecisionRequest>)`, and `agent.Plan(request)` reports the reads, canvas
widths and prompt lengths without running anything.

What a read is: the questions become a system prompt; the answer is a template `0:A\n1:no\n2:0` after
Gemma's empty thought block, rounded up to a 16-token canvas; the answer slots are filled with seeded
noise; one denoising step later, each slot's distribution over its allowed label ids is the answer.
Requests that cannot be answered that way - a label that is not one token, a template wider than the
canvas, a prompt beyond the context, more than djev's limits - fail with `DecisionSchemaException` before
any model work. A read that comes back without complete, valid evidence fails with
`DecisionBackendException`; nothing is filled in.

`IDecisionReader` is the whole model dependency (a tokenizer, the chat prompt, a batched one-step read), so
the agent runs over anything that can provide it; `DiffusionGemmaReader` is the in-process one.

## Benchmarking against JevBench

`JevBenchSet` reads [jevbench](https://github.com/fstandhartinger/jevbench)'s public decisions from a
checkout (hash-checked against its manifest), and `JevBenchRunner` sends each one the way jevbench's djev
adapter does and scores it with jevbench's rules: accuracy per tier, chance-corrected Intelligence,
Calibration from hard-tier ECE and gold distributions, p50/p95 Speed with jevbench's endpoint adjustment,
and Cost per 1,000 decisions from the input tokens read. The receipt is a public-items estimate, never a
JevBench Score - part of every tier is held out.

```csharp
var set = JevBenchSet.Load("/path/to/jevbench");
JevBenchReport report = await new JevBenchRunner(agent).RunAsync(set);
File.WriteAllText("jevbench.json", report.ToJson());
```

The command-line harness is [`benchmarks/DiffusionDecisionBench`](../benchmarks/DiffusionDecisionBench/README.md).

## The JSON canvas (open-jev)

Typed JSON decisions on a block-diffusion model. The model is never asked to write an
answer that is then parsed — it is handed a canvas that can only hold answers, and the
readout picks among them.

```csharp
var reader = new DiffusionGemmaReader(model);
var predictor = new StructuredPredictor(reader);

var request = new StructuredRequest
{
    Id = "ticket-1",
    Document = "I was charged twice for the same order. Please refund the duplicate.",
    Questions = new Dictionary<string, StructuredQuestion>
    {
        ["refund_requested"] = StructuredQuestion.Boolean("Does the customer ask for a refund?"),
        ["department"] = StructuredQuestion.Choice("Which team handles this?", "billing", "technical", "sales"),
        ["severity"] = StructuredQuestion.Score("How severe is this?", "low", "medium", "high"),
    },
};

StructuredPrediction prediction = await predictor.PredictAsync(request);
// prediction.Json    -> {"refund_requested": true, "department": "billing", "severity": 1}
// prediction.Values  -> { refund_requested = true, department = "billing", severity = 1 }
// prediction.Fields["department"].OptionProbabilities -> one score per allowed value
```

Every prediction is a complete, valid member of the request's allowed language. There is
no JSON repair, no retry and no second pass, and one denoising step answers every
question about a document at once.

### How the canvas works

1. **Tokenize every allowed answer as a complete JSON document.** Not a schema — the
   finite set of documents the answer may be.
2. **Pin what they all agree on.** The braces, the quoted keys, the separators sit at the
   same token positions in every candidate, so they are held for the whole denoise. A
   pinned position contributes no entropy and settles immediately: it is the cheap form
   of a logits mask that allows exactly one token there.
3. **Leave the rest free.** Only the positions where candidates disagree denoise, and they
   denoise unrestricted — nothing constrains them until the readout.
4. **Read out over the allowed tokens.** Per field, walk its free positions left to right
   and keep the candidates carrying the highest-scoring allowed token at each. The first
   disagreeing position does most of the work; later ones resolve what survives, and a
   unique suffix is forced.
5. **Decode and check.** If the selected document does not parse back to an allowed
   answer, the canvas was wrong and it throws — it is never repaired.

Candidates of a field are padded to a common width with a single whitespace token, so a
field occupies the same span whichever answer wins and the fields compose without
enumerating their Cartesian product.

### What this does not do

- **Questions are not isolated from one another.** Questions on one canvas share
  attention, so an answer can shift depending on what it was asked alongside.
- **Questions that do not fit one canvas are split** across several and merged back, which
  changes what each one attends to. `PlanCanvasCounts` says where that happens.
- **`OptionProbabilities` is uncalibrated.** It is a softmax over the allowed tokens of a
  field's first discriminating position, from the same diffusion pass. It ranks; it is not
  a truth probability, and values sharing a token at that position share its mass.
- **The `<user_text>` delimiters are formatting, not a security boundary.** A document that
  writes `</user_text>` is not stopped from doing so.
- **Later readout decisions are not autoregressive.** They use the scores of the same
  diffusion pass, not likelihoods conditioned on the prefix just chosen.

### Benchmarking

`StructuredBenchmark` measures what a decision service is actually judged on — accuracy
against references, throughput, and cost — and writes a receipt that carries its own
caveats.

```csharp
var report = await new StructuredBenchmark(predictor).RunAsync(cases, new StructuredBenchmarkOptions
{
    BatchSizes = new[] { 64, 32, 16, 8, 4, 2, 1 },
    Steps = 1,
    Repeats = 3,
    Warmups = 1,
    DeviceUsdPerSecond = 0.001097,
    PricingSource = "https://example/pricing (checked YYYY-MM-DD)",
});
File.WriteAllText("results/structured.json", report.ToJson());
```

The sweep starts at the largest batch and halves on an out-of-memory failure, so what it
reports is the best batch the device actually held rather than one chosen in advance —
and the sizes that did not fit are listed in `failures` rather than dropped. Warmup passes
absorb first-request overhead; model loading is never inside the timings. `Repeats` also
gives `inconsistent_repeated_documents`: the same document, the same seed, answered more
than one way.

Anything that is not an out-of-memory failure propagates. A wrong schema must not be
silently recorded as a batch that was too big.

The receipt's field names match [open-jev](https://github.com/theolivenbaum/open-jev)'s
(`documents_per_second`, `gpu_usd_per_1000_judgments`, …) so runs can be put side by side.
Read `cost_scope` before quoting a dollar figure: it is device time over the timed passes
only, and not an invoice.

Accuracy accounts for every question asked, not just the ones it could score.
`StructuredBenchmarkCase.Unscored` carries why a question has no reference, and the receipt
reports the counts: a dataset where some questions were never adjudicated, or where the
adjudicators tied, is not the same as one where they all agree, and `scored` plus `unscored`
has to add up to the questions asked. `Baseline` carries another system's answers and is
scored on exactly the same questions — a comparison between two systems measured on
different subsets is not one.

### Evaluating against open-jev's public set

open-jev ships the public evaluation materials it measured on. `OpenJevEvalSet` reads them
into benchmark cases, resolving references the same way open-jev does — the consensus of a
question's adjudication sets, averaged, scored only where one value leads outright:

```csharp
var set = OpenJevEvalSet.Load("/path/to/open-jev");
var report = await new StructuredBenchmark(predictor).RunAsync(set.Cases);
```

The data is **not vendored here**. It is third-party evaluation material with its own
rights, published with source URLs and hashes in open-jev's `manifest.json`; point the
loader at a checkout. `SourceSha256` records what was actually read, so a receipt can name
its inputs.

As a check that the two agree on what is being measured, the importer reproduces open-jev's
published shape exactly: 408 questions, 337 scorable, 54 without a reference, 17 tied, and
its saved baseline at 90.8% overall (78.8 / 89.1 / 97.0 / 80.8 by workflow). `OpenJevEvalSetTests`
asserts those numbers against a checkout named by `TS_OPEN_JEV_DIR`.

One difference to know about: open-jev puts a question's criteria into the prompt as the raw
JSON the dataset stores (a `{label: description}` object, or a list for a score). This
library aligns `Criteria` with `Options` and prints them as parallel arrays, so the prompts
are worded differently even though the allowed values are identical.

### Relationship to open-jev

open-jev is the Python research harness for this idea on DiffusionGemma. This library is
the same construction in the TensorSharp engine: a pinned JSON template, free answer slots,
and a constrained final readout over the allowed tokens.

Two differences are worth knowing:

- open-jev applies a `[canvas, vocab]` logits mask each step. TensorSharp pins positions
  instead — the same effect at the positions that matter, without materializing the mask.
- open-jev pads the canvas to the model's full width with EOS and always denoises it.
  TensorSharp does the same by default, and `CanvasFit = Tight` instead compiles the canvas
  to the width the answers need and runs the forward there (see below).

### Canvas width

By default the answers are padded out to the model's served canvas with end-of-sequence
tokens — the block the model was trained on. `CanvasFit = Tight` compiles the canvas to
the width the answers actually need, plus one terminator, and the forward runs at that
width:

```csharp
await predictor.PredictAsync(request, new StructuredPredictOptions
{
    CanvasFit = JsonCanvasFit.Tight,
});
```

Attention, the MoE and the lm_head all scale with the canvas width, so for a short typed
answer this is most of the cost. `PlanCanvasWidths` says what either fit would run at,
without running anything, and the benchmark receipt records `canvas_widths` and
`canvas_fit` beside the throughput — a documents-per-second figure cannot be read without
knowing which canvas bought it.

**It is not free.** A 32-wide canvas is a 32-token block, not a 256-token block with 224
positions ignored. The model sees a shorter block than the one it was trained on, and the
answer can move. Benchmark both fits on your own data and compare the accuracy, not only
the throughput.

### Using a different model

`IStructuredReader` is the whole dependency: a tokenizer, the canvas geometry, and a way
to denoise a batch of seeded canvases and report the scores at their free positions. It
takes a batch because that is where the throughput is — a canvas is one forward pass, so
several cost barely more than one. `DiffusionGemmaReader` is the implementation for a
loaded DiffusionGemma GGUF; it holds the model's compute lock for the block, so hand it
the whole batch rather than calling it once per request.
