// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Structured
{
    /// <summary>One benchmark case: a request, and what its answers should have been.</summary>
    public sealed class StructuredBenchmarkCase
    {
        public required StructuredRequest Request { get; init; }

        /// <summary>Reference answers by question key. A key that is absent is not scored - a case with
        /// no reference at all still contributes to throughput, never to accuracy.</summary>
        public IReadOnlyDictionary<string, object?> Expected { get; init; } =
            new Dictionary<string, object?>();

        /// <summary>
        /// Why a question has no reference, by key: a dataset where some questions were never
        /// adjudicated, or where the adjudicators tied, is not the same as one where they all agree.
        /// Counted in the receipt so the scored total is auditable rather than a silent subset.
        /// </summary>
        public IReadOnlyDictionary<string, string> Unscored { get; init; } =
            new Dictionary<string, string>();

        /// <summary>Another system's answers on the same questions, by key. Scored alongside, so a
        /// receipt says how the run did AND how the system it is being compared to did on exactly the
        /// same subset.</summary>
        public IReadOnlyDictionary<string, object?> Baseline { get; init; } =
            new Dictionary<string, object?>();

        /// <summary>Groups cases in the report (a workflow, a dataset, an experiment).</summary>
        public string Workflow { get; init; } = "all";
    }

    /// <summary>How the sweep is run.</summary>
    public sealed class StructuredBenchmarkOptions
    {
        /// <summary>Batch sizes to try, largest first. The sweep stops at the first one that fits, so
        /// this reports the best batch the device holds rather than every batch.</summary>
        public IReadOnlyList<int> BatchSizes { get; init; } = new[] { 64, 32, 16, 8, 4, 2, 1 };

        /// <summary>Denoising steps per canvas.</summary>
        public int Steps { get; init; } = 1;

        public int Seed { get; init; }

        /// <summary>Passes over the whole case set that are timed. More than one exposes variance and
        /// whether repeated documents answer consistently.</summary>
        public int Repeats { get; init; } = 3;

        /// <summary>Untimed passes first, so the measured ones do not pay for first-request overhead
        /// (kernel selection, allocator growth, weight paging).</summary>
        public int Warmups { get; init; } = 1;

        /// <summary>Device cost per second, for the cost estimate. It is an estimate of GPU time only -
        /// not CPU, RAM, storage, or anything outside the measured calls, and not an invoice.</summary>
        public double DeviceUsdPerSecond { get; init; }

        /// <summary>Where the rate came from, carried into the receipt.</summary>
        public string? PricingSource { get; init; }

        /// <summary>How wide a canvas the answers are compiled onto. <see cref="JsonCanvasFit.Tight"/>
        /// runs the forward at the width the answers need; compare its accuracy against
        /// <see cref="JsonCanvasFit.ServedCanvas"/>, not only its throughput.</summary>
        public JsonCanvasFit CanvasFit { get; init; } = JsonCanvasFit.ServedCanvas;
    }

    /// <summary>What one batch size achieved.</summary>
    public sealed class StructuredBenchmarkSummary
    {
        [JsonPropertyName("requested_batch_size")] public required int RequestedBatchSize { get; init; }
        [JsonPropertyName("actual_batch_sizes")] public required IReadOnlyList<int> ActualBatchSizes { get; init; }
        [JsonPropertyName("documents_measured")] public required int DocumentsMeasured { get; init; }
        [JsonPropertyName("judgments_measured")] public required int JudgmentsMeasured { get; init; }
        [JsonPropertyName("canvases_measured")] public required int CanvasesMeasured { get; init; }
        [JsonPropertyName("documents_per_second")] public required double DocumentsPerSecond { get; init; }
        [JsonPropertyName("judgments_per_second")] public required double JudgmentsPerSecond { get; init; }
        [JsonPropertyName("amortized_ms_per_document")] public required double AmortizedMsPerDocument { get; init; }
        [JsonPropertyName("median_batch_latency_ms")] public required double MedianBatchLatencyMs { get; init; }
        [JsonPropertyName("gpu_usd_per_1000_documents")] public required double? UsdPer1000Documents { get; init; }
        [JsonPropertyName("gpu_usd_per_1000_judgments")] public required double? UsdPer1000Judgments { get; init; }

        /// <summary>Every prediction is a complete member of its allowed language by construction, so this
        /// is expected to equal <c>judgments_measured</c>; it is reported because a shortfall would mean
        /// the canvas, not the model, was wrong.</summary>
        [JsonPropertyName("valid_answers")] public required int ValidAnswers { get; init; }

        /// <summary>Documents whose repeated passes did not answer identically.</summary>
        [JsonPropertyName("inconsistent_repeated_documents")] public required int InconsistentRepeatedDocuments { get; init; }

        [JsonPropertyName("scored")] public required int Scored { get; init; }
        [JsonPropertyName("correct")] public required int Correct { get; init; }
        [JsonPropertyName("accuracy")] public required double? Accuracy { get; init; }

        /// <summary>Questions with no usable reference, by reason. Scored + these is every question asked.</summary>
        [JsonPropertyName("unscored")] public required IReadOnlyDictionary<string, int> Unscored { get; init; }

        /// <summary>How the compared system did on the same scored questions, when one was supplied.</summary>
        [JsonPropertyName("baseline_correct")] public required int? BaselineCorrect { get; init; }
        [JsonPropertyName("baseline_accuracy")] public required double? BaselineAccuracy { get; init; }

        /// <summary>How often this run and the compared system gave the same answer, right or wrong.</summary>
        [JsonPropertyName("agreement_with_baseline")] public required double? AgreementWithBaseline { get; init; }

        [JsonPropertyName("by_workflow")] public required IReadOnlyDictionary<string, WorkflowScore> ByWorkflow { get; init; }
    }

    /// <summary>Accuracy within one workflow.</summary>
    public sealed class WorkflowScore
    {
        [JsonPropertyName("scored")] public required int Scored { get; init; }
        [JsonPropertyName("correct")] public required int Correct { get; init; }
        [JsonPropertyName("accuracy")] public required double? Accuracy { get; init; }
        [JsonPropertyName("baseline_correct")] public required int? BaselineCorrect { get; init; }
        [JsonPropertyName("baseline_accuracy")] public required double? BaselineAccuracy { get; init; }
    }

    /// <summary>A batch size that did not fit, and what it said on the way out.</summary>
    public sealed class StructuredBenchmarkFailure
    {
        [JsonPropertyName("requested_batch_size")] public required int RequestedBatchSize { get; init; }
        [JsonPropertyName("error")] public required string Error { get; init; }
    }

    /// <summary>The receipt: what was measured, on what, and with which caveats.</summary>
    public sealed class StructuredBenchmarkReport
    {
        [JsonPropertyName("recorded_at")] public required DateTimeOffset RecordedAt { get; init; }
        [JsonPropertyName("cases")] public required int Cases { get; init; }
        [JsonPropertyName("questions")] public required int Questions { get; init; }
        [JsonPropertyName("canvases_per_pass")] public required int CanvasesPerPass { get; init; }

        /// <summary>The canvas width the forward ran at, widest first. Under the served-canvas fit this is
        /// the model's canvas whatever the answers need; under the tight fit it is what they needed.</summary>
        [JsonPropertyName("canvas_widths")] public required IReadOnlyList<int> CanvasWidths { get; init; }
        [JsonPropertyName("canvas_fit")] public required string CanvasFit { get; init; }
        [JsonPropertyName("split_cases")] public required int SplitCases { get; init; }
        [JsonPropertyName("steps")] public required int Steps { get; init; }
        [JsonPropertyName("seed")] public required int Seed { get; init; }
        [JsonPropertyName("repeats")] public required int Repeats { get; init; }
        [JsonPropertyName("warmups")] public required int Warmups { get; init; }
        [JsonPropertyName("device_usd_per_second")] public required double? DeviceUsdPerSecond { get; init; }
        [JsonPropertyName("pricing_source")] public required string? PricingSource { get; init; }
        [JsonPropertyName("summaries")] public required IReadOnlyList<StructuredBenchmarkSummary> Summaries { get; init; }
        [JsonPropertyName("failures")] public required IReadOnlyList<StructuredBenchmarkFailure> Failures { get; init; }
        [JsonPropertyName("cost_scope")] public required string CostScope { get; init; }
        [JsonPropertyName("scope")] public required string Scope { get; init; }

        public string ToJson() => JsonSerializer.Serialize(this, ReportJson);

        private static readonly JsonSerializerOptions ReportJson = new()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        };
    }

    /// <summary>
    /// Accuracy, throughput and cost for typed decisions, measured the way a decision service is actually
    /// used: whole documents in, one canvas per document, batched.
    ///
    /// The sweep starts at the largest batch and halves on an out-of-memory failure, so what it reports is
    /// the best batch the device held rather than a batch chosen in advance. Timings are warm: the warmup
    /// passes absorb first-request overhead, and model loading is never inside them.
    /// </summary>
    public sealed class StructuredBenchmark
    {
        private readonly StructuredPredictor _predictor;

        public StructuredBenchmark(StructuredPredictor predictor) =>
            _predictor = predictor ?? throw new ArgumentNullException(nameof(predictor));

        public async Task<StructuredBenchmarkReport> RunAsync(
            IReadOnlyList<StructuredBenchmarkCase> cases,
            StructuredBenchmarkOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(cases);
            if (cases.Count == 0) throw new ArgumentException("Nothing to measure.", nameof(cases));
            options ??= new StructuredBenchmarkOptions();
            if (options.Repeats < 1) throw new ArgumentException("Need at least one timed pass.", nameof(options));

            IReadOnlyList<StructuredRequest> requests = cases.Select(c => c.Request).ToList();
            IReadOnlyList<int> canvasCounts = _predictor.PlanCanvasCounts(requests, options.CanvasFit);

            var summaries = new List<StructuredBenchmarkSummary>();
            var failures = new List<StructuredBenchmarkFailure>();
            foreach (int batchSize in options.BatchSizes)
            {
                try
                {
                    summaries.Add(await MeasureAsync(
                        cases, requests, canvasCounts, batchSize, options, cancellationToken)
                        .ConfigureAwait(false));
                    // The first size that fits is the answer; larger ones already failed.
                    break;
                }
                catch (Exception ex) when (IsOutOfMemory(ex))
                {
                    failures.Add(new StructuredBenchmarkFailure
                    {
                        RequestedBatchSize = batchSize,
                        Error = ex.Message,
                    });
                }
            }

            return new StructuredBenchmarkReport
            {
                RecordedAt = DateTimeOffset.UtcNow,
                Cases = cases.Count,
                Questions = cases.Sum(c => c.Request.Questions.Count),
                CanvasesPerPass = canvasCounts.Sum(),
                CanvasWidths = _predictor.PlanCanvasWidths(requests, options.CanvasFit)
                    .Distinct().OrderDescending().ToList(),
                CanvasFit = options.CanvasFit.ToString(),
                SplitCases = canvasCounts.Count(n => n > 1),
                Steps = options.Steps,
                Seed = options.Seed,
                Repeats = options.Repeats,
                Warmups = options.Warmups,
                DeviceUsdPerSecond = options.DeviceUsdPerSecond > 0 ? options.DeviceUsdPerSecond : null,
                PricingSource = options.PricingSource,
                Summaries = summaries,
                Failures = failures,
                CostScope = "Device-time estimate over the timed passes only. Excludes model loading, "
                    + "host CPU, memory and storage, and any failed batch size. Not an invoice.",
                Scope = "One canvas per document where the questions fit; oversized requests are split "
                    + "and merged, so questions on one canvas attend to each other. Questions are not "
                    + "isolated from one another.",
            };
        }

        private async Task<StructuredBenchmarkSummary> MeasureAsync(
            IReadOnlyList<StructuredBenchmarkCase> cases,
            IReadOnlyList<StructuredRequest> requests,
            IReadOnlyList<int> canvasCounts,
            int batchSize,
            StructuredBenchmarkOptions options,
            CancellationToken cancellationToken)
        {
            var predictOptions = new StructuredPredictOptions
            {
                Steps = options.Steps,
                Seed = options.Seed,
                BatchSize = batchSize,
                CanvasFit = options.CanvasFit,
            };

            for (int i = 0; i < options.Warmups; i++)
                await _predictor.PredictAsync(requests, predictOptions, cancellationToken).ConfigureAwait(false);

            var latencies = new List<double>(options.Repeats);
            var passes = new List<IReadOnlyList<StructuredPrediction>>(options.Repeats);
            double totalSeconds = 0;
            for (int repeat = 0; repeat < options.Repeats; repeat++)
            {
                var watch = Stopwatch.StartNew();
                IReadOnlyList<StructuredPrediction> predictions = await _predictor
                    .PredictAsync(requests, predictOptions, cancellationToken).ConfigureAwait(false);
                watch.Stop();
                totalSeconds += watch.Elapsed.TotalSeconds;
                latencies.Add(watch.Elapsed.TotalMilliseconds);
                passes.Add(predictions);
            }

            int documents = cases.Count * options.Repeats;
            int judgments = cases.Sum(c => c.Request.Questions.Count) * options.Repeats;
            int canvases = canvasCounts.Sum() * options.Repeats;
            double? usd = options.DeviceUsdPerSecond > 0
                ? options.DeviceUsdPerSecond * totalSeconds : null;

            // Determinism across the timed passes: the same document, the same seed, the same answer.
            int inconsistent = 0;
            for (int i = 0; i < cases.Count; i++)
            {
                string first = passes[0][i].Json;
                if (passes.Any(pass => pass[i].Json != first)) inconsistent++;
            }

            Scoreboard board = Score(cases, passes[0]);

            return new StructuredBenchmarkSummary
            {
                RequestedBatchSize = batchSize,
                ActualBatchSizes = ActualBatches(canvasCounts.Sum(), batchSize),
                DocumentsMeasured = documents,
                JudgmentsMeasured = judgments,
                CanvasesMeasured = canvases,
                DocumentsPerSecond = documents / totalSeconds,
                JudgmentsPerSecond = judgments / totalSeconds,
                AmortizedMsPerDocument = 1000 * totalSeconds / documents,
                MedianBatchLatencyMs = Median(latencies),
                UsdPer1000Documents = usd is { } u ? 1000 * u / documents : null,
                UsdPer1000Judgments = usd is { } v ? 1000 * v / judgments : null,
                ValidAnswers = passes.Sum(pass => pass.Sum(p => p.Fields.Count)),
                InconsistentRepeatedDocuments = inconsistent,
                Scored = board.Scored,
                Correct = board.Correct,
                Accuracy = board.Scored > 0 ? (double)board.Correct / board.Scored : null,
                Unscored = board.Unscored,
                BaselineCorrect = board.HasBaseline ? board.BaselineCorrect : null,
                BaselineAccuracy = board.HasBaseline && board.Scored > 0
                    ? (double)board.BaselineCorrect / board.Scored : null,
                AgreementWithBaseline = board.HasBaseline && board.Scored > 0
                    ? (double)board.Agreed / board.Scored : null,
                ByWorkflow = board.ByWorkflow,
            };
        }

        /// <summary>Running totals while a pass is compared against its references.</summary>
        private sealed record Scoreboard(
            int Scored,
            int Correct,
            int BaselineCorrect,
            int Agreed,
            bool HasBaseline,
            IReadOnlyDictionary<string, int> Unscored,
            IReadOnlyDictionary<string, WorkflowScore> ByWorkflow);

        /// <summary>
        /// Compare one pass against the references. A question with no reference is not scored but is
        /// counted under why, so the scored total can be audited against the questions asked rather than
        /// being a silent subset. Where a baseline was supplied it is scored on exactly the same
        /// questions - a comparison between two systems measured on different subsets is not one.
        /// Scoring the first timed pass keeps the number independent of how many passes were run.
        /// </summary>
        private static Scoreboard Score(
            IReadOnlyList<StructuredBenchmarkCase> cases, IReadOnlyList<StructuredPrediction> predictions)
        {
            var scored = new Dictionary<string, int>();
            var correct = new Dictionary<string, int>();
            var baselineCorrect = new Dictionary<string, int>();
            var unscored = new Dictionary<string, int>();
            int agreed = 0;
            bool hasBaseline = false;

            for (int i = 0; i < cases.Count; i++)
            {
                StructuredBenchmarkCase benchmarkCase = cases[i];
                string workflow = benchmarkCase.Workflow;
                foreach ((string key, StructuredFieldAnswer answer) in predictions[i].Fields)
                {
                    if (!benchmarkCase.Expected.TryGetValue(key, out object? expected))
                    {
                        string reason = benchmarkCase.Unscored.GetValueOrDefault(key, "no_reference");
                        unscored[reason] = unscored.GetValueOrDefault(reason) + 1;
                        continue;
                    }
                    scored[workflow] = scored.GetValueOrDefault(workflow) + 1;
                    string ours = JsonValue.Serialize(answer.Value);
                    if (ours == JsonValue.Serialize(expected))
                        correct[workflow] = correct.GetValueOrDefault(workflow) + 1;

                    if (!benchmarkCase.Baseline.TryGetValue(key, out object? baseline)) continue;
                    hasBaseline = true;
                    string theirs = JsonValue.Serialize(baseline);
                    if (theirs == JsonValue.Serialize(expected))
                        baselineCorrect[workflow] = baselineCorrect.GetValueOrDefault(workflow) + 1;
                    if (theirs == ours) agreed++;
                }
            }

            var byWorkflow = scored.ToDictionary(kv => kv.Key, kv => new WorkflowScore
            {
                Scored = kv.Value,
                Correct = correct.GetValueOrDefault(kv.Key),
                Accuracy = kv.Value > 0 ? (double)correct.GetValueOrDefault(kv.Key) / kv.Value : null,
                BaselineCorrect = hasBaseline ? baselineCorrect.GetValueOrDefault(kv.Key) : null,
                BaselineAccuracy = hasBaseline && kv.Value > 0
                    ? (double)baselineCorrect.GetValueOrDefault(kv.Key) / kv.Value : null,
            });
            return new Scoreboard(
                scored.Values.Sum(), correct.Values.Sum(), baselineCorrect.Values.Sum(), agreed,
                hasBaseline, unscored, byWorkflow);
        }

        private static IReadOnlyList<int> ActualBatches(int canvases, int batchSize)
        {
            var sizes = new SortedSet<int>();
            for (int offset = 0; offset < canvases; offset += batchSize)
                sizes.Add(Math.Min(batchSize, canvases - offset));
            return sizes.ToList();
        }

        private static double Median(List<double> values)
        {
            var sorted = values.Order().ToList();
            int mid = sorted.Count / 2;
            return sorted.Count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        }

        /// <summary>The device ran out of memory. Backends report this differently - a managed
        /// <see cref="OutOfMemoryException"/>, or a native allocator's message - so both are recognised,
        /// and anything else is a real failure that must not be mistaken for a batch that was too big.</summary>
        private static bool IsOutOfMemory(Exception ex)
        {
            for (Exception? e = ex; e is not null; e = e.InnerException)
            {
                if (e is OutOfMemoryException) return true;
                if (e.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase)) return true;
            }
            return false;
        }
    }
}
