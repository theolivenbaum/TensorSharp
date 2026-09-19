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
        [JsonPropertyName("by_workflow")] public required IReadOnlyDictionary<string, WorkflowScore> ByWorkflow { get; init; }
    }

    /// <summary>Accuracy within one workflow.</summary>
    public sealed class WorkflowScore
    {
        [JsonPropertyName("scored")] public required int Scored { get; init; }
        [JsonPropertyName("correct")] public required int Correct { get; init; }
        [JsonPropertyName("accuracy")] public required double? Accuracy { get; init; }
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
            IReadOnlyList<int> canvasCounts = _predictor.PlanCanvasCounts(requests);

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

            (int scored, int correct, Dictionary<string, WorkflowScore> byWorkflow) = Score(cases, passes[0]);

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
                Scored = scored,
                Correct = correct,
                Accuracy = scored > 0 ? (double)correct / scored : null,
                ByWorkflow = byWorkflow,
            };
        }

        /// <summary>Compare one pass against the references. A question with no reference is not scored;
        /// scoring the first timed pass keeps the number independent of how many passes were run.</summary>
        private static (int Scored, int Correct, Dictionary<string, WorkflowScore> ByWorkflow) Score(
            IReadOnlyList<StructuredBenchmarkCase> cases, IReadOnlyList<StructuredPrediction> predictions)
        {
            var scored = new Dictionary<string, int>();
            var correct = new Dictionary<string, int>();
            for (int i = 0; i < cases.Count; i++)
            {
                StructuredBenchmarkCase benchmarkCase = cases[i];
                foreach ((string key, StructuredFieldAnswer answer) in predictions[i].Fields)
                {
                    if (!benchmarkCase.Expected.TryGetValue(key, out object? expected)) continue;
                    string workflow = benchmarkCase.Workflow;
                    scored[workflow] = scored.GetValueOrDefault(workflow) + 1;
                    if (JsonValue.Serialize(expected) == JsonValue.Serialize(answer.Value))
                        correct[workflow] = correct.GetValueOrDefault(workflow) + 1;
                }
            }
            var byWorkflow = scored.ToDictionary(kv => kv.Key, kv => new WorkflowScore
            {
                Scored = kv.Value,
                Correct = correct.GetValueOrDefault(kv.Key),
                Accuracy = kv.Value > 0 ? (double)correct.GetValueOrDefault(kv.Key) / kv.Value : null,
            });
            return (scored.Values.Sum(), correct.Values.Sum(), byWorkflow);
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
