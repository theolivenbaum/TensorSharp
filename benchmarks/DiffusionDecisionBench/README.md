# DiffusionDecisionBench

Typed DiffusionGemma decisions from the command line, the way [djev](https://github.com/Davipar/djev-dev) reads
them - the same commands as Laya's CLI (`predict`, `bench`, `presets`), plus a replay of
[jevbench](https://github.com/fstandhartinger/jevbench)'s public decisions into a scored receipt.

```bash
dotnet build benchmarks/DiffusionDecisionBench -c Release
B=benchmarks/DiffusionDecisionBench/bin/Release/net10.0/DiffusionDecisionBench
M=models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

## Predict

```bash
$B predict --model $M --backend ggmlcuda --preset triage --text "I was charged twice and nobody answers"

$B predict --model $M --backend ggmlcuda \
    --question 'urgent=noul:Is this time critical?' \
    --question 'team=choice:Who handles this?|billing,support,security' \
    --question 'impact=score:How bad is it?|none,minor,major,outage' \
    --state-file ticket.json --samples 2 --diagnostics

$B predict --model $M --request request.json    # a djev POST /v1/request body, verbatim
```

The answer is printed as djev's response body; compile time, model time, canvas width and input tokens go to
stderr.

## Bench

```bash
$B bench --model $M --backend ggmlcuda --preset triage --text "…" --iterations 10
```

One untimed warm-up, then `N` timed runs of the same request, the median, and milliseconds per question -
Laya's `bench`. Use `--isolation independent` to see what one read per question costs.

## JevBench

```bash
git clone https://github.com/fstandhartinger/jevbench /path/to/jevbench

$B jevbench --model $M --backend ggmlcuda --jevbench /path/to/jevbench --out results/jevbench.json
$B jevbench ... --splits easy,original --limit 40          # a smoke run
$B jevbench ... --pacing batched --batch-size 32           # throughput instead of latency
$B plan --model $M --jevbench /path/to/jevbench            # reads, canvas widths, prompt tokens; nothing is run
```

Each public decision is sent the way jevbench's djev adapter sends it - the state, and the question under the
key `decision`, with djev's default options (one sample, seed 0, joint) unless you pass others - and scored
with jevbench's rules (`jevbench/scoring.py`, `metrics.py`, `composite_v13.py`):

| receipt field | what it is |
|---|---|
| `tiers` | accuracy per tier (`easy`, `standard` = `original.jsonl`, `hard`), with each tier's uniform-guessing baseline and chance-corrected score |
| `intelligence` | weighted chance-corrected accuracy over the tiers run (hard 30, easy 14, standard 28, renormalised); `intelligence_published_chance` uses jevbench's published baselines, which include held-out items |
| `calibration` | hard tier: `100·(1 − ECE/0.5)`, averaged with `100·(1 − mean TVD)` to the gold distributions |
| `p50_s`, `p95_s`, `speed` | serial, in-process latency; Speed adjusted as jevbench adjusts self-hosted endpoints (`--endpoint gpu`: ×2 + 0.15 s) |
| `usd_per_1000_decisions`, `cost` | input tokens read × `--price-per-m` (default djev's announced $0.035/M, output free); `--device-usd-per-hour` adds a device-time price |
| `public_subset_score` | the v1.3 geometric mean over those four axes - an estimate on the public items, **not** a JevBench Score |
| `outcomes` | every decision: prediction, distribution, latency, tokens, canvas width, error |

Every split file's SHA-256 is compared with jevbench's `datasets/manifest.json` and recorded in `sources`.
The data is not vendored; point the harness at a checkout.

Serial pacing (the default) is jevbench's protocol: one request at a time, each timed alone, so p50/p95 are
latencies. Batched pacing hands every decision to the agent at once and reports amortized time per decision,
which is a throughput figure; the receipt says which one it is.

## Without a model

`--mock` swaps the model for a stand-in reader (word tokens, hashed scores) so the whole pipeline - compile,
batch, validate, score, write the receipt - runs anywhere:

```bash
$B jevbench --mock --jevbench /path/to/jevbench --out /tmp/mock.json
```

Its accuracy is chance, as it should be. It checks the harness, never the model.

## Serving

The same decisions are served over HTTP by `TensorSharp.Server.Host` with a DiffusionGemma GGUF loaded:
`POST /v1/request` takes and returns djev's bodies, so jevbench's djev adapter can point at it
(`DjevAdapter(endpoint="http://localhost:5000")`, any API key).
