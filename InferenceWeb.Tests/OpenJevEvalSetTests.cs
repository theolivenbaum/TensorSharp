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
// The importer for open-jev's public evaluation set. The point of these is that the numbers below are
// open-jev's own published ones: if this reads the data or resolves a reference differently, a receipt
// produced here could not be put beside one produced there, however well the model did.
//
// The data is third-party and not vendored, so these run against a checkout named by TS_OPEN_JEV_DIR:
//   git clone https://github.com/theolivenbaum/open-jev
//   TS_OPEN_JEV_DIR=/path/to/open-jev dotnet test
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured;
using TensorSharp.Structured.Evals;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class OpenJevEvalSetTests
{
    private const string EnvEvalDir = "TS_OPEN_JEV_DIR";
    private const string What = "the open-jev public evaluation set";
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string GgufPattern = "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private readonly ITestOutputHelper _output;
    public OpenJevEvalSetTests(ITestOutputHelper output) => _output = output;

    private static OpenJevEvalSet.Result Load() =>
        OpenJevEvalSet.Load(System.Environment.GetEnvironmentVariable(EnvEvalDir)!);

    [EvalDataFact(EnvEvalDir, What)]
    public void TheSetLoadsWithTheQuestionCountsItsAuthorsPublished()
    {
        var set = Load();
        foreach (OpenJevWorkflowCounts w in set.Counts)
            _output.WriteLine($"[open-jev] {w.Workflow}: {w.Questions} questions, {w.Scorable} scorable, " +
                $"{w.MissingReference} without a reference, {w.ReferenceTie} tied");

        // open-jev replays 408 questions and scores 337 of them; the other 71 are questions nobody
        // adjudicated (54) or where the adjudicators tied (17).
        Assert.Equal(408, set.Questions);
        Assert.Equal(337, set.Scorable);
        Assert.Equal(54, set.Counts.Sum(c => c.MissingReference));
        Assert.Equal(17, set.Counts.Sum(c => c.ReferenceTie));

        var scorable = set.Counts.ToDictionary(c => c.Workflow, c => c.Scorable);
        Assert.Equal(52, scorable["agent_trace_observability"]);
        Assert.Equal(92, scorable["customer_service"]);
        Assert.Equal(167, scorable["invoice_processing"]);
        Assert.Equal(26, scorable["security_incidents"]);
    }

    [EvalDataFact(EnvEvalDir, What)]
    public void TheSavedBaselineScoresWhatItsAuthorsReported()
    {
        // Resolving a reference is the whole comparison: consensus over the adjudication sets, averaged,
        // and only where one value leads outright. Score the set's own saved answers against the
        // references this importer resolved and the published accuracy has to come back out.
        var set = Load();
        var correct = new Dictionary<string, int>();
        var scored = new Dictionary<string, int>();
        foreach (StructuredBenchmarkCase c in set.Cases)
        {
            foreach ((string key, object? expected) in c.Expected)
            {
                scored[c.Workflow] = scored.GetValueOrDefault(c.Workflow) + 1;
                if (Equals(c.Baseline[key], expected))
                    correct[c.Workflow] = correct.GetValueOrDefault(c.Workflow) + 1;
            }
        }

        double Accuracy(string workflow) => 100.0 * correct[workflow] / scored[workflow];
        foreach (string workflow in scored.Keys.OrderBy(x => x))
            _output.WriteLine($"[open-jev] {workflow}: baseline {Accuracy(workflow):F1}%");

        Assert.Equal(78.8, Accuracy("agent_trace_observability"), 1);
        Assert.Equal(89.1, Accuracy("customer_service"), 1);
        Assert.Equal(97.0, Accuracy("invoice_processing"), 1);
        Assert.Equal(80.8, Accuracy("security_incidents"), 1);
        Assert.Equal(90.8, 100.0 * correct.Values.Sum() / scored.Values.Sum(), 1);
    }

    [EvalDataFact(EnvEvalDir, What)]
    public void EveryCaseIsAnAnswerableRequest_WithTypedReferences()
    {
        var set = Load();
        Assert.All(set.Cases, c => c.Request.Validate());

        // A reference and a saved answer are the question's own allowed values, not the strings the
        // dataset writes them as - otherwise every comparison would be against something unanswerable.
        foreach (StructuredBenchmarkCase c in set.Cases)
        {
            foreach ((string key, object? expected) in c.Expected)
                Assert.Contains(expected, c.Request.Questions[key].Options);
            foreach ((string key, object? saved) in c.Baseline)
                Assert.Contains(saved, c.Request.Questions[key].Options);
            // Scored and unscored partition the node's questions; nothing is quietly dropped.
            Assert.Equal(
                c.Request.Questions.Keys.OrderBy(x => x),
                c.Expected.Keys.Concat(c.Unscored.Keys).OrderBy(x => x));
        }

        var kinds = set.Cases
            .SelectMany(c => c.Request.Questions.Values)
            .GroupBy(q => q.Kind!)
            .ToDictionary(g => g.Key, g => g.Count());
        _output.WriteLine($"[open-jev] question types: {string.Join(", ", kinds.Select(k => $"{k.Key}={k.Value}"))}");
        Assert.Equal(264, kinds["noul"]);
        Assert.Equal(117, kinds["choice"]);
        Assert.Equal(27, kinds["score"]);

        // The receipt has to be able to name what it read.
        Assert.Equal(4, set.SourceSha256.Count);
        Assert.All(set.SourceSha256.Values, sha => Assert.Equal(64, sha.Length));
    }

    [EvalDataFact(EnvEvalDir, What)]
    public void ScoreQuestionsKeepTheirScale_AndNoulQuestionsTheirWording()
    {
        var set = Load();
        var score = set.Cases
            .SelectMany(c => c.Request.Questions.Values)
            .First(q => q.Kind == "score");
        var noul = set.Cases
            .SelectMany(c => c.Request.Questions.Values)
            .First(q => q.Kind == "noul");

        // A score is answered by a zero-based index, and its criteria say what each index means.
        Assert.Equal(Enumerable.Range(0, score.Options.Count).Cast<object?>(), score.Options);
        Assert.Equal(score.Options.Count, score.Criteria!.Count);

        // A yes/no keeps its two descriptions in the order its options are in, so the prompt lines up.
        Assert.Equal(new object?[] { false, true }, noul.Options);
        Assert.Equal(2, noul.Criteria!.Count);
    }

    // The whole point of importing the set: the same questions, scored the same way, so a receipt
    // produced here can be put beside open-jev's. Needs both the data and the weights.
    [EvalDataModelFact(EnvEvalDir, What, EnvModelDir, GgufPattern)]
    public async Task ReplayingTheSetProducesAComparableReceipt()
    {
        string dir = System.Environment.GetEnvironmentVariable(EnvModelDir)!;
        string path = TestGates.FindGguf(dir, GgufPattern)!;
        using var model = (DiffusionGemmaModel)ModelBase.Create(path, TestGates.PreferredTestBackend);

        var set = Load();
        var benchmark = new StructuredBenchmark(
            new StructuredPredictor(new DiffusionGemmaReader(model)));
        StructuredBenchmarkReport report = await benchmark.RunAsync(set.Cases,
            new StructuredBenchmarkOptions
            {
                BatchSizes = new[] { 16, 8, 4, 2, 1 },
                Steps = 1,
                Repeats = 1,
                Warmups = 1,
            });

        _output.WriteLine(report.ToJson());
        StructuredBenchmarkSummary summary = Assert.Single(report.Summaries);

        // Scored on exactly the questions open-jev scores, with its baseline measured on the same ones.
        Assert.Equal(set.Scorable, summary.Scored);
        Assert.Equal(54, summary.Unscored["missing_reference"]);
        Assert.Equal(17, summary.Unscored["reference_tie"]);
        Assert.Equal(90.8, 100 * summary.BaselineAccuracy!.Value, 1);
        _output.WriteLine($"[open-jev] ours {100 * summary.Accuracy!.Value:F1}% vs baseline " +
            $"{100 * summary.BaselineAccuracy.Value:F1}%, agreeing " +
            $"{100 * summary.AgreementWithBaseline!.Value:F1}% of the time");
    }
}
