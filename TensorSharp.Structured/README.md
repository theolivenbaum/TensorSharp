# TensorSharp.Structured

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

## How the canvas works

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

## Benchmarking

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

## Evaluating against open-jev's public set

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

## Relationship to open-jev

open-jev is the Python research harness for this idea on DiffusionGemma. This library is
the same construction in the TensorSharp engine: a pinned JSON template, free answer slots,
and a constrained final readout over the allowed tokens.

Two differences are worth knowing:

- open-jev applies a `[canvas, vocab]` logits mask each step. TensorSharp pins positions
  instead — the same effect at the positions that matter, without materializing the mask.
- open-jev pads the canvas to the model's full width with EOS and always denoises it.
  TensorSharp does the same by default, and `CanvasFit = Tight` instead compiles the canvas
  to the width the answers need and runs the forward there (see below).

## Canvas width

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

## Using a different model

`IStructuredReader` is the whole dependency: a tokenizer, the canvas geometry, and a way
to denoise a batch of seeded canvases and report the scores at their free positions. It
takes a batch because that is where the throughput is — a canvas is one forward pass, so
several cost barely more than one. `DiffusionGemmaReader` is the implementation for a
loaded DiffusionGemma GGUF; it holds the model's compute lock for the block, so hand it
the whole batch rather than calling it once per request.
