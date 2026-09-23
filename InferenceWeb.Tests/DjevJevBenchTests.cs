// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// The JevBench harness: jevbench's scoring rules as pure functions (no data needed), the loader and the
// runner against a jevbench checkout named by TS_JEVBENCH_DIR, and - with TS_TEST_MODEL_DIR as well - a
// replay of the public decisions through a real DiffusionGemma checkpoint.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Structured.Decisions;
using TensorSharp.Structured.Evals;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class DjevJevBenchTests
{
    private const string EnvJevBench = "TS_JEVBENCH_DIR";
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string GgufPattern = "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private readonly ITestOutputHelper _output;

    public DjevJevBenchTests(ITestOutputHelper output) => _output = output;

    // ---- Scoring rules, no data --------------------------------------------------------------------------

    [Fact]
    public void PublishedTierChancesMatchJevbench()
    {
        // composite_v13.TIER_CHANCES, from the published option-count histograms.
        Assert.Equal((18 / 2.0 + 13 / 4.0 + 41 / 5.0) / 72, JevBenchScoring.PublishedTierChances["easy"], 12);
        Assert.Equal((68 / 2.0 + 78 / 9.0) / 146, JevBenchScoring.PublishedTierChances["judge"], 12);
    }

    [Fact]
    public void AxesFollowComposite13()
    {
        Assert.Equal(100, JevBenchScoring.SpeedPoint(0.1), 9);
        Assert.Equal(80, JevBenchScoring.SpeedPoint(1.0), 9);
        Assert.Equal(0.3, JevBenchScoring.AdjustedLatency(0.075, "gpu"), 12);   // ×2 + 0.15 s on an own server
        Assert.Equal(0.075, JevBenchScoring.AdjustedLatency(0.075, "api"), 12);
        Assert.Equal(100, JevBenchScoring.Cost(0.001), 9);
        Assert.Equal(70, JevBenchScoring.Cost(0.01), 9);
        Assert.Throws<ArgumentOutOfRangeException>(() => JevBenchScoring.Cost(0));
        Assert.Equal(80, JevBenchScoring.Calibration(0.1, null)!.Value, 9);
        Assert.Equal(70, JevBenchScoring.Calibration(0.1, 0.4)!.Value, 9);
        Assert.Equal(50, JevBenchScoring.ChanceCorrected(0.75, 0.5), 9);
        Assert.Equal(0, JevBenchScoring.ChanceCorrected(0.25, 0.5), 9);
        // Geometric mean, and the (I/50)^2 penalty below 50 Intelligence.
        Assert.Equal(64, JevBenchScoring.Composite(64, 64, 64, 64), 9);
        Assert.Equal(40 * 0.64, JevBenchScoring.Composite(40, 40, 40, 40), 9);
        // Renormalised over the tiers that ran.
        Assert.Equal(50, JevBenchScoring.Intelligence(
            new Dictionary<string, double> { ["easy"] = 0.75 }, new Dictionary<string, double> { ["easy"] = 0.5 })!.Value, 9);
    }

    [Fact]
    public void DistributionsAreValidatedAndScoredLikeJevbench()
    {
        var task = new JevBenchTask
        {
            Id = "t", Tier = "easy", Split = "easy", Family = "f", State = "s",
            QuestionJson = new System.Text.Json.Nodes.JsonObject(),
            Question = Question.Choice("?", "a", "b"), Labels = new[] { "a", "b" }, Expected = "b",
        };
        Assert.True(JevBenchScoring.Score(new Dictionary<string, double> { ["a"] = 0.4, ["b"] = 0.6 }, task).Correct);
        // Ties break towards the lexicographically smallest label.
        Assert.Equal("a", JevBenchScoring.Score(new Dictionary<string, double> { ["b"] = 0.5, ["a"] = 0.5 }, task).Predicted);
        // Inside the rounding band a distribution is rescaled; outside it is invalid and wrong.
        var rounded = JevBenchScoring.Score(new Dictionary<string, double> { ["a"] = 0.4, ["b"] = 0.61 }, task);
        Assert.True(rounded.Valid && rounded.Renormalized && !rounded.StrictValid);
        var broken = JevBenchScoring.Score(new Dictionary<string, double> { ["a"] = 0.4, ["b"] = 0.7 }, task);
        Assert.False(broken.Valid);
        Assert.False(broken.Correct);
        Assert.False(JevBenchScoring.Score(new Dictionary<string, double> { ["a"] = 1.0 }, task).Valid);
        Assert.Equal(0.5, JevBenchScoring.Brier(new Dictionary<string, double> { ["a"] = 0.5, ["b"] = 0.5 }, "b", task.Labels), 12);
        // Ten answers at 0.9 confidence, eight right: one bin, |0.8 - 0.9|.
        var pairs = Enumerable.Range(0, 10).Select(i => (0.9, i < 8)).ToList();
        Assert.Equal(0.1, JevBenchScoring.Ece(pairs)!.Value, 12);
        Assert.Equal(2.5, JevBenchScoring.Percentile(new double[] { 1, 2, 3, 4 }, 0.5));
    }

    [Fact]
    public void NoulAnswersBecomeYesNoDistributions()
    {
        var answer = new Answer { Type = QuestionType.Noul, Noul = 0.8 };
        var distribution = JevBenchRunner.Distribution(answer);
        Assert.Equal(0.8, distribution["yes"], 12);
        Assert.Equal(0.2, distribution["no"], 12);
    }

    // ---- Against a jevbench checkout ----------------------------------------------------------------------

    [EvalDataFact(EnvJevBench, "a jevbench checkout")]
    public void LoadsThePublicSetAndMatchesTheManifest()
    {
        JevBenchSet set = JevBenchSet.Load(Environment.GetEnvironmentVariable(EnvJevBench)!);
        Assert.Equal(231, set.Tasks.Count);
        Assert.Equal(48, set.Tasks.Count(t => t.Tier == "easy"));
        Assert.Equal(72, set.Tasks.Count(t => t.Tier == "standard"));   // original.jsonl is the standard tier
        Assert.Equal(111, set.Tasks.Count(t => t.Tier == "hard"));
        Assert.All(set.Sources, s => Assert.True(s.MatchesManifest, $"{s.Split} does not match the manifest"));
        Assert.Equal(10, set.Tasks.Count(t => t.GoldProbs is not null));
        // jevbench stores criteria key-sorted and labels in authoring order; djev receives the sorted
        // criteria, so the option order differs from the label order but the set is the same.
        Assert.All(set.Tasks, t => Assert.Equal(t.Labels.Order(), t.Question.Labels.Order()));
        Assert.All(set.Tasks.Where(t => t.Expected is not null), t => Assert.Contains(t.Expected!, t.Labels));
    }

    [EvalDataFact(EnvJevBench, "a jevbench checkout")]
    public async Task ReplaysEveryPublicDecisionIntoAReceipt()
    {
        JevBenchSet set = JevBenchSet.Load(Environment.GetEnvironmentVariable(EnvJevBench)!);
        var reader = new DjevDecisionTests.RecordingReader();
        using var agent = new DiffusionAgent(reader);
        JevBenchReport report = await new JevBenchRunner(agent).RunAsync(set, new JevBenchRunOptions { Warmups = 0 });

        Assert.Equal(231, report.Decisions);
        Assert.Equal(0, report.Failed);
        Assert.Equal(231, report.Valid);
        Assert.Equal(231, reader.Reads.Count);
        Assert.Equal(new[] { "easy", "hard", "standard" }, report.Tiers.Keys.Order());
        Assert.All(report.Outcomes, o => Assert.Equal(1.0, o.Probs!.Values.Sum(), 9));
        Assert.NotNull(report.Intelligence);
        Assert.NotNull(report.Calibration);
        Assert.Contains("\"public_subset_score\"", report.ToJson());
    }

    // ---- Against a real checkpoint ------------------------------------------------------------------------

    private DiffusionAgent LoadAgent()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir)!, GgufPattern)
            ?? Environment.GetEnvironmentVariable(EnvModelDir)!;
        _output.WriteLine($"loading {Path.GetFileName(path)} on {TestGates.PreferredTestBackend}");
        return DiffusionAgent.Load(path, TestGates.PreferredTestBackend);
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void GemmaTokenizerKeepsEveryDjevLabelOneToken()
    {
        using DiffusionAgent agent = LoadAgent();
        // The widest schemas djev supports: every choice label, every score digit, both noul labels.
        var set = new QuestionSet()
            .Add("wide", Question.Choice("Pick one", Enumerable.Range(0, DecisionContracts.MaxChoiceOptions).Select(i => $"o{i}").ToArray()))
            .Add("score", Question.Score("Rate", Enumerable.Range(0, 10).Select(i => (object?)$"level {i}").ToArray()))
            .Add("noul", Question.Noul("True?"));
        CompiledDecisionSchema schema = agent.Compile(set);
        Assert.Equal(DecisionContracts.MaxChoiceOptions, schema.Slots[0].TokenIds.Count);
        int width = agent.Compile(Presets.Triage()).CanvasWidth;
        Assert.True(width is 16 or 32, $"triage compiled to a {width}-token canvas");
        foreach (string preset in Presets.Names) agent.Compile(Presets.ByName(preset));
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void AnswersTriageInOneRead()
    {
        using DiffusionAgent agent = LoadAgent();
        DecisionResult result = agent.SystemOne("I was charged twice for the same order and nobody answers my emails.",
            Presets.Triage(), new DecisionOptions { Diagnostics = true });
        _output.WriteLine(result.ToJsonString(indented: true));
        Assert.Equal(5, result.Answers.Count);
        Assert.Equal(1, result.Diagnostics!["physical_reads"]!.GetValue<int>());
        Assert.InRange(result["refund_requested"].Noul!.Value, 0, 1);
        Assert.Contains(result["intent"].Choice, Presets.Triage()["intent"].Labels);
    }

    [EvalDataModelFact(EnvJevBench, "a jevbench checkout", EnvModelDir, GgufPattern)]
    public async Task ReplaysJevBenchThroughTheModel()
    {
        JevBenchSet set = JevBenchSet.Load(Environment.GetEnvironmentVariable(EnvJevBench)!);
        using DiffusionAgent agent = LoadAgent();
        int? limit = int.TryParse(Environment.GetEnvironmentVariable("TS_JEVBENCH_LIMIT"), out int l) ? l : null;
        JevBenchReport report = await new JevBenchRunner(agent).RunAsync(set, new JevBenchRunOptions { Limit = limit });
        foreach ((string tier, JevBenchTierSummary summary) in report.Tiers)
            _output.WriteLine($"{tier}: {summary.Correct}/{summary.Scorable} ({summary.Accuracy:P1})");
        _output.WriteLine($"intelligence {report.Intelligence:F1}, calibration {report.Calibration:F1}, p50 {report.P50S:F3} s, " +
                          $"public-subset score {report.PublicSubsetScore:F1}");
        Assert.Equal(0, report.Failed);
        Assert.Equal(report.Decisions, report.Valid);
    }
}
