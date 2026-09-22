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
using System.Linq;

namespace TensorSharp.Structured.Evals
{
    /// <summary>
    /// JevBench's scoring rules, as pure functions (<c>jevbench/scoring.py</c>, <c>metrics.py</c> and
    /// <c>composite_v13.py</c>), so a receipt produced here is computed the way the published board is.
    /// </summary>
    public static class JevBenchScoring
    {
        /// <summary>The pre-registered distribution-sum tolerance.</summary>
        public const double SumTolerance = 1e-3;

        /// <summary>The band inside which a distribution is treated as rounded and rescaled.</summary>
        public const double RenormTolerance = 2e-2;

        /// <summary>Intelligence weights per tier. Missing tiers are renormalised away.</summary>
        public static readonly IReadOnlyDictionary<string, double> TierWeights = new Dictionary<string, double>
        {
            ["easy"] = 0.14, ["standard"] = 0.28, ["judge"] = 0.28, ["hard"] = 0.30,
        };

        /// <summary>The published per-tier uniform-guessing baselines (v1.3), from the option-count
        /// histograms of every frozen item, held-out ones included.</summary>
        public static readonly IReadOnlyDictionary<string, double> PublishedTierChances = new Dictionary<string, double>
        {
            ["easy"] = Chance(new Dictionary<int, int> { [2] = 18, [4] = 13, [5] = 41 }),
            ["standard"] = Chance(new Dictionary<int, int> { [2] = 32, [4] = 40, [5] = 12, [6] = 12 }),
            ["judge"] = Chance(new Dictionary<int, int> { [2] = 68, [9] = 78 }),
            ["hard"] = Chance(new Dictionary<int, int> { [2] = 77, [3] = 26, [4] = 73, [5] = 38, [6] = 6 }),
        };

        private static double Chance(IReadOnlyDictionary<int, int> counts) =>
            counts.Sum(c => c.Value / (double)c.Key) / counts.Values.Sum();

        /// <summary>What a distribution scores against a task (<c>score_task</c>).</summary>
        public sealed record TaskScore(bool Valid, bool StrictValid, bool Renormalized, string? Predicted,
            bool? Correct, IReadOnlyDictionary<string, double>? Probs, string? Error);

        /// <summary>
        /// Validate a distribution against the exact label set and score it. Keys must match exactly,
        /// values must be finite and in [0, 1] and sum to 1 within <see cref="SumTolerance"/>; inside
        /// <see cref="RenormTolerance"/> it is rescaled, outside it is invalid and wrong. Ties in the argmax
        /// break towards the lexicographically smallest label.
        /// </summary>
        public static TaskScore Score(IReadOnlyDictionary<string, double> probs, JevBenchTask task)
        {
            var want = new HashSet<string>(task.Labels, StringComparer.Ordinal);
            if (!want.SetEquals(probs.Keys))
                return new TaskScore(false, false, false, null, task.Expected is null ? null : false, null, "label keys mismatch");
            if (probs.Values.Any(v => !double.IsFinite(v) || v < 0 || v > 1))
                return new TaskScore(false, false, false, null, task.Expected is null ? null : false, null, "probability out of range");
            double total = probs.Values.Sum();
            bool strict = Math.Abs(total - 1) <= SumTolerance;
            if (!strict && (Math.Abs(total - 1) > RenormTolerance || total <= 0))
                return new TaskScore(false, false, false, null, task.Expected is null ? null : false, null,
                    $"probs sum to {total}");
            var clean = strict
                ? probs.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal)
                : probs.ToDictionary(p => p.Key, p => p.Value / total, StringComparer.Ordinal);
            string predicted = Argmax(clean);
            return new TaskScore(true, strict, !strict, predicted,
                task.Expected is null ? null : predicted == task.Expected, clean, null);
        }

        /// <summary>Deterministic argmax, ties broken by the lexicographically smallest label.</summary>
        public static string Argmax(IReadOnlyDictionary<string, double> probs)
        {
            string? best = null;
            double bestP = -1;
            foreach (string key in probs.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                if (probs[key] > bestP) { best = key; bestP = probs[key]; }
            }
            return best!;
        }

        /// <summary>Multi-class Brier over the exact label set; binary questions use the 2-class form.</summary>
        public static double Brier(IReadOnlyDictionary<string, double> probs, string expected, IReadOnlyList<string> labels) =>
            labels.Sum(l => Math.Pow(probs.GetValueOrDefault(l) - (l == expected ? 1.0 : 0.0), 2));

        /// <summary>Top-label ECE over (confidence, correct) pairs, 10 equal-width bins.</summary>
        public static double? Ece(IReadOnlyList<(double Confidence, bool Correct)> pairs, int bins = 10)
        {
            if (pairs.Count == 0) return null;
            var n = new int[bins];
            var conf = new double[bins];
            var correct = new int[bins];
            foreach ((double c0, bool ok) in pairs)
            {
                double c = Math.Clamp(c0, 0, 1);
                int b = Math.Min((int)(c * bins), bins - 1);
                n[b]++;
                conf[b] += c;
                if (ok) correct[b]++;
            }
            double ece = 0;
            for (int b = 0; b < bins; b++)
                if (n[b] > 0) ece += (double)n[b] / pairs.Count * Math.Abs((double)correct[b] / n[b] - conf[b] / n[b]);
            return ece;
        }

        /// <summary>Total variation distance between two distributions over <paramref name="labels"/>.</summary>
        public static double Tvd(IReadOnlyDictionary<string, double> p, IReadOnlyDictionary<string, double> q,
            IEnumerable<string> labels) =>
            0.5 * labels.Sum(l => Math.Abs(p.GetValueOrDefault(l) - q.GetValueOrDefault(l)));

        /// <summary>jevbench's percentile: linear interpolation between closest ranks.</summary>
        public static double? Percentile(IReadOnlyList<double> values, double q)
        {
            if (values.Count == 0) return null;
            var sorted = values.Order().ToArray();
            double k = (sorted.Length - 1) * q;
            int f = (int)Math.Floor(k), c = (int)Math.Ceiling(k);
            return f == c ? sorted[f] : sorted[f] * (c - k) + sorted[c] * (k - f);
        }

        private static double Clamp(double x) => Math.Clamp(x, 0, 100);

        public static double ChanceCorrected(double accuracy, double chance) => Clamp(100 * (accuracy - chance) / (1 - chance));

        /// <summary>Weighted chance-corrected tier accuracy; tiers not run are renormalised away.</summary>
        public static double? Intelligence(IReadOnlyDictionary<string, double> tierAccuracy,
            IReadOnlyDictionary<string, double> tierChance)
        {
            double score = 0, weight = 0;
            foreach ((string tier, double w) in TierWeights)
            {
                if (!tierAccuracy.TryGetValue(tier, out double accuracy)) continue;
                score += w * ChanceCorrected(accuracy, tierChance[tier]);
                weight += w;
            }
            return weight > 0 ? score / weight : null;
        }

        /// <summary>Calibration: 100·(1 − ECE/0.5), averaged with 100·(1 − mean TVD) where gold
        /// distributions exist.</summary>
        public static double? Calibration(double? ece, double? meanTvd)
        {
            if (ece is null) return null;
            double eceScore = Math.Max(0, 100 * (1 - ece.Value / 0.5));
            return meanTvd is null ? eceScore : (eceScore + 100 * (1 - meanTvd.Value)) / 2;
        }

        /// <summary>jevbench's latency adjustment: production APIs as measured; anything else ×2, and our
        /// own servers (<c>gpu</c>, <c>cpu</c>) +0.15 s more. An assumption about load, not a measurement.</summary>
        public static double AdjustedLatency(double seconds, string endpointKind) => endpointKind switch
        {
            "api" => seconds,
            "gpu" or "cpu" => seconds * 2 + 0.15,
            _ => seconds * 2,
        };

        public static double SpeedPoint(double seconds) => Clamp(100 - 20 * Math.Log10(seconds / 0.1));

        public static double Speed(double p50, double p95, string endpointKind) =>
            (SpeedPoint(AdjustedLatency(p50, endpointKind)) + SpeedPoint(AdjustedLatency(p95, endpointKind))) / 2;

        /// <summary>Cost axis from dollars per 1,000 decisions. A missing or zero price is an error, never
        /// an automatic 100.</summary>
        public static double Cost(double usdPer1000Decisions)
        {
            if (!(usdPer1000Decisions > 0))
                throw new ArgumentOutOfRangeException(nameof(usdPer1000Decisions), "every system needs a positive price");
            return Clamp(100 - 30 * Math.Log10(usdPer1000Decisions / 0.001));
        }

        /// <summary>The four-axis geometric mean, with the near-chance penalty below 50 Intelligence.</summary>
        public static double Composite(double? intelligence, double? calibration, double? speed, double? cost)
        {
            double[] axes = { intelligence ?? 0, calibration ?? 0, speed ?? 0, cost ?? 0 };
            double baseScore = Math.Exp(axes.Sum(a => 0.25 * Math.Log(Math.Max(a, 1))));
            double multiplier = intelligence is null or >= 50 ? 1 : Math.Pow(Math.Max(intelligence.Value, 0) / 50, 2);
            return baseScore * multiplier;
        }
    }
}
