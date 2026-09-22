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
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Structured.Decisions;

namespace TensorSharp.Structured.Evals
{
    /// <summary>How requests are sent.</summary>
    public enum JevBenchPacing
    {
        /// <summary>One request at a time, each timed on its own - jevbench's protocol, and the only pacing
        /// whose p50/p95 mean latency.</summary>
        Serial,

        /// <summary>All requests at once, batched by the agent. Latency is then amortized time per
        /// decision, a throughput figure; it is labelled as such and not comparable to serial p50.</summary>
        Batched,
    }

    public sealed record JevBenchRunOptions
    {
        /// <summary>Per-request options, sent with every decision. jevbench's djev adapter sends none, so
        /// djev's defaults (one sample, seed 0, joint) are the like-for-like setting.</summary>
        public DecisionOptions Decision { get; init; } = DecisionOptions.Default;

        public JevBenchPacing Pacing { get; init; } = JevBenchPacing.Serial;

        /// <summary>Untimed requests first, so kernel selection and allocator growth are not timed.</summary>
        public int Warmups { get; init; } = 2;

        /// <summary>Run only the first tasks of the set (for smoke runs). Null runs every task.</summary>
        public int? Limit { get; init; }

        /// <summary>Price per million input tokens for the Cost axis; output tokens are free, as djev
        /// prices them. Default: djev's announced $0.035/M.</summary>
        public double UsdPerMillionInputTokens { get; init; } = 0.035;

        public string PricingSource { get; init; } = "djev announced price, $0.035 per million input tokens, output free";

        /// <summary>Device cost per hour. When set, the receipt also prices the run by device time.</summary>
        public double? DeviceUsdPerHour { get; init; }

        /// <summary>jevbench's endpoint kind for the latency adjustment: <c>gpu</c>/<c>cpu</c> (own
        /// server, ×2 + 0.15 s), <c>api</c> (production, unadjusted), anything else ×2.</summary>
        public string EndpointKind { get; init; } = "gpu";

        public string SystemName { get; init; } = "TensorSharp DiffusionGemma (djev read)";
    }

    /// <summary>One decision's outcome.</summary>
    public sealed record JevBenchOutcome
    {
        [JsonPropertyName("id")] public required string Id { get; init; }
        [JsonPropertyName("tier")] public required string Tier { get; init; }
        [JsonPropertyName("family")] public required string Family { get; init; }
        [JsonPropertyName("type")] public required string Type { get; init; }
        [JsonPropertyName("ok")] public required bool Ok { get; init; }
        [JsonPropertyName("valid")] public bool Valid { get; init; }
        [JsonPropertyName("expected")] public string? Expected { get; init; }
        [JsonPropertyName("predicted")] public string? Predicted { get; init; }
        [JsonPropertyName("correct")] public bool? Correct { get; init; }
        [JsonPropertyName("confidence")] public double? Confidence { get; init; }
        [JsonPropertyName("probs")] public IReadOnlyDictionary<string, double>? Probs { get; init; }
        [JsonPropertyName("latency_s")] public double LatencyS { get; init; }
        [JsonPropertyName("input_tokens")] public int InputTokens { get; init; }
        [JsonPropertyName("canvas_tokens")] public int? CanvasTokens { get; init; }
        [JsonPropertyName("error")] public string? Error { get; init; }
    }

    public sealed record JevBenchTierSummary
    {
        [JsonPropertyName("n")] public required int N { get; init; }
        [JsonPropertyName("scorable")] public required int Scorable { get; init; }
        [JsonPropertyName("correct")] public required int Correct { get; init; }
        [JsonPropertyName("accuracy")] public required double? Accuracy { get; init; }
        [JsonPropertyName("chance_items")] public required double Chance { get; init; }
        [JsonPropertyName("chance_corrected")] public required double? ChanceCorrected { get; init; }
    }

    /// <summary>The receipt.</summary>
    public sealed record JevBenchReport
    {
        [JsonPropertyName("recorded_at")] public required DateTimeOffset RecordedAt { get; init; }
        [JsonPropertyName("system")] public required string System { get; init; }
        [JsonPropertyName("model")] public required string Model { get; init; }
        [JsonPropertyName("protocol")] public string? Protocol { get; init; }
        [JsonPropertyName("sources")] public required IReadOnlyList<JevBenchSource> Sources { get; init; }
        [JsonPropertyName("options")] public required JsonObject Options { get; init; }
        [JsonPropertyName("pacing")] public required string Pacing { get; init; }
        [JsonPropertyName("decisions")] public required int Decisions { get; init; }
        [JsonPropertyName("failed")] public required int Failed { get; init; }
        [JsonPropertyName("valid")] public required int Valid { get; init; }
        [JsonPropertyName("accuracy")] public required double? Accuracy { get; init; }
        [JsonPropertyName("tiers")] public required IReadOnlyDictionary<string, JevBenchTierSummary> Tiers { get; init; }
        [JsonPropertyName("families")] public required IReadOnlyDictionary<string, JevBenchTierSummary> Families { get; init; }
        [JsonPropertyName("intelligence")] public required double? Intelligence { get; init; }
        [JsonPropertyName("intelligence_published_chance")] public required double? IntelligencePublishedChance { get; init; }
        [JsonPropertyName("ece_hard")] public required double? EceHard { get; init; }
        [JsonPropertyName("mean_tvd_hard")] public required double? MeanTvdHard { get; init; }
        [JsonPropertyName("calibration")] public required double? Calibration { get; init; }
        [JsonPropertyName("brier_mean")] public required double? BrierMean { get; init; }
        [JsonPropertyName("p50_s")] public required double? P50S { get; init; }
        [JsonPropertyName("p95_s")] public required double? P95S { get; init; }
        [JsonPropertyName("mean_latency_s")] public required double? MeanLatencyS { get; init; }
        [JsonPropertyName("endpoint_kind")] public required string EndpointKind { get; init; }
        [JsonPropertyName("speed")] public required double? Speed { get; init; }
        [JsonPropertyName("decisions_per_second")] public required double DecisionsPerSecond { get; init; }
        [JsonPropertyName("mean_input_tokens")] public required double MeanInputTokens { get; init; }
        [JsonPropertyName("usd_per_1000_decisions")] public required double? UsdPer1000Decisions { get; init; }
        [JsonPropertyName("pricing_source")] public required string PricingSource { get; init; }
        [JsonPropertyName("device_usd_per_1000_decisions")] public required double? DeviceUsdPer1000Decisions { get; init; }
        [JsonPropertyName("cost")] public required double? Cost { get; init; }
        [JsonPropertyName("public_subset_score")] public required double? PublicSubsetScore { get; init; }
        [JsonPropertyName("canvas_widths")] public required IReadOnlyList<int> CanvasWidths { get; init; }
        [JsonPropertyName("caveats")] public required IReadOnlyList<string> Caveats { get; init; }
        [JsonPropertyName("outcomes")] public required IReadOnlyList<JevBenchOutcome> Outcomes { get; init; }

        public string ToJson() => JsonSerializer.Serialize(this, Json);

        private static readonly JsonSerializerOptions Json = new()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        };
    }

    /// <summary>
    /// Runs a <see cref="JevBenchSet"/> through a <see cref="DiffusionAgent"/> the way jevbench runs its
    /// djev row: each task is the state plus one question under the key <c>decision</c>, no options beyond
    /// the configured ones, one request at a time. The answer is mapped to a distribution exactly as
    /// jevbench's djev adapter maps it (Noul <c>P(yes)</c> to <c>{yes, no}</c>; Choice and Score
    /// probabilities as returned) and scored with jevbench's rules.
    /// </summary>
    public sealed class JevBenchRunner
    {
        private readonly DiffusionAgent _agent;

        public JevBenchRunner(DiffusionAgent agent) => _agent = agent ?? throw new ArgumentNullException(nameof(agent));

        /// <summary>jevbench's djev adapter: a typed answer as the distribution the scorer reads.</summary>
        public static IReadOnlyDictionary<string, double> Distribution(Answer answer) => answer.Type == QuestionType.Noul
            ? new Dictionary<string, double> { ["yes"] = answer.Noul!.Value, ["no"] = 1.0 - answer.Noul!.Value }
            : answer.Probabilities!.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);

        public async Task<JevBenchReport> RunAsync(JevBenchSet set, JevBenchRunOptions? options = null,
            IProgress<JevBenchOutcome>? progress = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(set);
            options ??= new JevBenchRunOptions();
            var tasks = (options.Limit is { } limit ? set.Tasks.Take(limit) : set.Tasks).ToList();
            if (tasks.Count == 0) throw new ArgumentException("the set has no tasks to run", nameof(set));

            for (int i = 0; i < Math.Min(options.Warmups, tasks.Count); i++)
            {
                try { await _agent.PredictAsync(tasks[i].ToRequest(options.Decision), cancellationToken).ConfigureAwait(false); }
                catch (DecisionSchemaException) { /* a warmup that cannot compile is reported by the timed pass */ }
            }

            var outcomes = new List<JevBenchOutcome>(tasks.Count);
            var wall = Stopwatch.StartNew();
            if (options.Pacing == JevBenchPacing.Serial)
            {
                foreach (JevBenchTask task in tasks)
                {
                    var watch = Stopwatch.StartNew();
                    DecisionResult? result = null;
                    string? error = null;
                    try
                    {
                        result = await _agent.PredictAsync(task.ToRequest(options.Decision), cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception ex) when (ex is DecisionSchemaException or DecisionBackendException)
                    {
                        error = ex.Message;
                    }
                    JevBenchOutcome outcome = Outcome(task, result, error, watch.Elapsed.TotalSeconds);
                    outcomes.Add(outcome);
                    progress?.Report(outcome);
                }
            }
            else
            {
                // Compile failures are per task; everything that compiles is batched together.
                var runnable = new List<JevBenchTask>();
                var failures = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (JevBenchTask task in tasks)
                {
                    try { _agent.Plan(task.ToRequest(options.Decision)); runnable.Add(task); }
                    catch (DecisionSchemaException ex) { failures[task.Id] = ex.Message; }
                }
                var watch = Stopwatch.StartNew();
                IReadOnlyList<DecisionResult> results = await _agent.PredictAsync(
                    runnable.Select(t => t.ToRequest(options.Decision)).ToList(), cancellationToken).ConfigureAwait(false);
                double perDecision = watch.Elapsed.TotalSeconds / Math.Max(1, runnable.Count);
                var byId = runnable.Select((t, i) => (t.Id, results[i])).ToDictionary(x => x.Id, x => x.Item2);
                foreach (JevBenchTask task in tasks)
                {
                    JevBenchOutcome outcome = byId.TryGetValue(task.Id, out DecisionResult? result)
                        ? Outcome(task, result, null, perDecision)
                        : Outcome(task, null, failures[task.Id], 0);
                    outcomes.Add(outcome);
                    progress?.Report(outcome);
                }
            }
            double wallSeconds = wall.Elapsed.TotalSeconds;
            return Report(set, tasks, outcomes, options, wallSeconds);
        }

        private static JevBenchOutcome Outcome(JevBenchTask task, DecisionResult? result, string? error, double seconds)
        {
            string type = Question.TypeName(task.Question.Type);
            if (result is null)
            {
                return new JevBenchOutcome
                {
                    Id = task.Id, Tier = task.Tier, Family = task.Family, Type = type, Ok = false,
                    Expected = task.Expected, Correct = task.Expected is null ? null : false,
                    LatencyS = seconds, Error = error,
                };
            }
            Answer answer = result[JevBenchSet.QuestionKey];
            IReadOnlyDictionary<string, double> probs = Distribution(answer);
            JevBenchScoring.TaskScore score = JevBenchScoring.Score(probs, task);
            return new JevBenchOutcome
            {
                Id = task.Id, Tier = task.Tier, Family = task.Family, Type = type, Ok = true,
                Valid = score.Valid, Expected = task.Expected, Predicted = score.Predicted, Correct = score.Correct,
                Confidence = score.Probs?.Values.Max(), Probs = score.Probs ?? probs,
                LatencyS = seconds, InputTokens = result.Usage.InputTokens,
                CanvasTokens = result.CanvasWidths.Count > 0 ? result.CanvasWidths.Max() : null, Error = score.Error,
            };
        }

        private static JevBenchTierSummary Summarize(IReadOnlyList<(JevBenchTask Task, JevBenchOutcome Outcome)> rows)
        {
            var scorable = rows.Where(r => r.Task.Expected is not null).ToList();
            int correct = scorable.Count(r => r.Outcome.Correct == true);
            double chance = scorable.Count > 0 ? scorable.Average(r => r.Task.Chance) : rows.Average(r => r.Task.Chance);
            double? accuracy = scorable.Count > 0 ? (double)correct / scorable.Count : null;
            return new JevBenchTierSummary
            {
                N = rows.Count, Scorable = scorable.Count, Correct = correct, Accuracy = accuracy, Chance = chance,
                ChanceCorrected = accuracy is { } a ? JevBenchScoring.ChanceCorrected(a, chance) : null,
            };
        }

        private JevBenchReport Report(JevBenchSet set, List<JevBenchTask> tasks, List<JevBenchOutcome> outcomes,
            JevBenchRunOptions options, double wallSeconds)
        {
            var rows = tasks.Zip(outcomes, (t, o) => (Task: t, Outcome: o)).ToList();
            var tiers = rows.GroupBy(r => r.Task.Tier).ToDictionary(g => g.Key, g => Summarize(g.ToList()));
            var families = rows.GroupBy(r => r.Task.Family).OrderBy(g => g.Key, StringComparer.Ordinal)
                .ToDictionary(g => g.Key, g => Summarize(g.ToList()));
            var scorable = rows.Where(r => r.Task.Expected is not null).ToList();

            var tierAccuracy = tiers.Where(t => t.Value.Accuracy is not null && JevBenchScoring.TierWeights.ContainsKey(t.Key))
                .ToDictionary(t => t.Key, t => t.Value.Accuracy!.Value);
            double? intelligence = JevBenchScoring.Intelligence(tierAccuracy, tiers.ToDictionary(t => t.Key, t => t.Value.Chance));
            double? intelligencePublished = JevBenchScoring.Intelligence(tierAccuracy, JevBenchScoring.PublishedTierChances);

            var hard = rows.Where(r => r.Task.Tier == "hard" && r.Task.Expected is not null).ToList();
            double? ece = JevBenchScoring.Ece(hard.Select(r => (r.Outcome.Valid ? r.Outcome.Confidence ?? 0 : 0,
                r.Outcome.Correct == true)).ToList());
            var gold = hard.Where(r => r.Task.GoldProbs is not null).ToList();
            double? meanTvd = gold.Count > 0
                ? gold.Average(r => r.Outcome.Probs is { } p && r.Outcome.Valid
                    ? JevBenchScoring.Tvd(p, r.Task.GoldProbs!, r.Task.GoldProbs!.Keys)
                    : 1.0)
                : null;
            double? calibration = JevBenchScoring.Calibration(ece, meanTvd);

            var briers = scorable.Where(r => r.Outcome.Valid && r.Outcome.Probs is not null)
                .Select(r => JevBenchScoring.Brier(r.Outcome.Probs!, r.Task.Expected!, r.Task.Labels)).ToList();

            var latencies = outcomes.Select(o => o.LatencyS).ToList();
            double? p50 = JevBenchScoring.Percentile(latencies, 0.5);
            double? p95 = JevBenchScoring.Percentile(latencies, 0.95);
            double? speed = p50 is { } a && p95 is { } b && a > 0 && b > 0
                ? JevBenchScoring.Speed(a, b, options.EndpointKind) : null;

            double meanInput = outcomes.Average(o => (double)o.InputTokens);
            double? usd = meanInput > 0 && options.UsdPerMillionInputTokens > 0
                ? meanInput * 1000 * options.UsdPerMillionInputTokens / 1_000_000 : null;
            double? deviceUsd = options.DeviceUsdPerHour is { } rate
                ? rate / 3600 * wallSeconds / outcomes.Count * 1000 : null;
            double? cost = usd is { } u ? JevBenchScoring.Cost(u) : null;

            var caveats = new List<string>
            {
                "Public items only: jevbench holds out part of every tier, so this is not a JevBench Score. " +
                "public_subset_score applies the v1.3 formula to the public items and is an estimate for comparison, not a rank.",
                "Intelligence uses the chance baseline of the items actually run; intelligence_published_chance uses the " +
                "published per-tier baselines, which include held-out items.",
                "Probabilities are relative to the allowed labels and confidence is entropy concentration; neither is calibrated.",
                options.Pacing == JevBenchPacing.Serial
                    ? $"Latency is measured in-process, one request at a time, and adjusted for endpoint kind '{options.EndpointKind}' " +
                      "as jevbench adjusts self-hosted endpoints; there is no network or HTTP in the timing."
                    : "Batched pacing: latency is amortized device time per decision, a throughput figure. p50/p95 and Speed are " +
                      "not comparable to jevbench's serial measurements.",
                "Cost prices input tokens as counted by this engine's tokenizer and chat template; it is a tariff estimate, not an invoice.",
            };
            if (set.Sources.Any(s => s.MatchesManifest == false))
                caveats.Add("At least one split file does not match jevbench's manifest hash; see sources.");

            return new JevBenchReport
            {
                RecordedAt = DateTimeOffset.UtcNow,
                System = options.SystemName,
                Model = _agent.ModelName,
                Protocol = set.Protocol,
                Sources = set.Sources,
                Options = options.Decision.ToJson(),
                Pacing = options.Pacing.ToString().ToLowerInvariant(),
                Decisions = outcomes.Count,
                Failed = outcomes.Count(o => !o.Ok),
                Valid = outcomes.Count(o => o.Valid),
                Accuracy = scorable.Count > 0 ? (double)scorable.Count(r => r.Outcome.Correct == true) / scorable.Count : null,
                Tiers = tiers,
                Families = families,
                Intelligence = intelligence,
                IntelligencePublishedChance = intelligencePublished,
                EceHard = ece,
                MeanTvdHard = meanTvd,
                Calibration = calibration,
                BrierMean = briers.Count > 0 ? briers.Average() : null,
                P50S = p50,
                P95S = p95,
                MeanLatencyS = latencies.Average(),
                EndpointKind = options.EndpointKind,
                Speed = speed,
                DecisionsPerSecond = wallSeconds > 0 ? outcomes.Count / wallSeconds : 0,
                MeanInputTokens = meanInput,
                UsdPer1000Decisions = usd,
                PricingSource = options.PricingSource,
                DeviceUsdPer1000Decisions = deviceUsd,
                Cost = cost,
                PublicSubsetScore = intelligence is null ? null
                    : JevBenchScoring.Composite(intelligence, calibration, speed, cost),
                CanvasWidths = outcomes.Where(o => o.CanvasTokens is not null).Select(o => o.CanvasTokens!.Value)
                    .Distinct().OrderDescending().ToList(),
                Caveats = caveats,
                Outcomes = outcomes,
            };
        }
    }
}
