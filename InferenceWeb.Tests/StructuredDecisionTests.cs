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
// Typed JSON decisions end to end, without a model: a character tokenizer compiles the canvases for real,
// and a scripted reader stands in for the denoise so the constrained readout, the canvas packing and the
// benchmark harness are exercised on their own terms. What a real model would change is which token wins
// a slot - which is exactly what the scripted reader controls here.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured;
using Xunit;

namespace InferenceWeb.Tests;

public class StructuredDecisionTests
{
    private const int CanvasLength = 96;

    /// <summary>One token per ASCII character. Round-trips exactly, so the canvas compiler's
    /// tokenization checks pass on their own merits rather than by construction.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int BosTokenId => 1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => 128;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) =>
            new(ids.Where(id => id != 0).Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.Add((byte)tokenId);
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    /// <summary>
    /// Stands in for the denoise. It reconstructs the canvas the predictor built - the prompt lists every
    /// question and its allowed values, which is exactly what the canvas was compiled from - and then
    /// scores the allowed tokens with the test's preferred answer ahead. So the readout, not the stub,
    /// decides what comes back.
    /// </summary>
    private sealed class ScriptedReader : IStructuredReader
    {
        private readonly Func<string, object?> _preferred;
        private readonly Func<int, bool>? _rejectBatch;

        public ScriptedReader(Func<string, object?> preferred, Func<int, bool>? rejectBatch = null)
        {
            _preferred = preferred;
            _rejectBatch = rejectBatch;
        }

        public int Reads { get; private set; }
        public List<int> BatchSizes { get; } = new();
        public List<int> Widths { get; } = new();

        public ITokenizer Tokenizer { get; } = new CharTokenizer();
        public int CanvasLength => StructuredDecisionTests.CanvasLength;
        public int VocabSize => 128;
        public int EosTokenId => 0;

        public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
            IReadOnlyList<StructuredRead> reads, CancellationToken cancellationToken = default)
        {
            if (_rejectBatch?.Invoke(reads.Count) == true)
                throw new InvalidOperationException("ggml_backend_alloc: out of memory");

            Reads++;
            BatchSizes.Add(reads.Count);
            var results = reads.Select(Answer).ToList();
            return Task.FromResult<IReadOnlyList<DiffusionReadResult>>(results);
        }

        private DiffusionReadResult Answer(StructuredRead read)
        {
            JsonCanvasLayout layout = JsonCanvasLayout.Compile(
                Tokenizer, QuestionsOnCanvas(read), CanvasLength, EosTokenId,
                // The read declares the width it wants the forward to run at; a tight canvas is shorter
                // than the served one, so the layout has to be rebuilt the same way.
                read.Options.CanvasWidth < CanvasLength ? JsonCanvasFit.Tight : JsonCanvasFit.ServedCanvas);
            Widths.Add(read.Options.CanvasWidth ?? CanvasLength);
            var prefer = layout.Questions.ToDictionary(
                q => q.Key,
                q => Math.Max(0, q.Value.Options.ToList().FindIndex(
                    o => JsonText(o) == JsonText(_preferred(q.Key)))));
            return new DiffusionReadResult(
                read.Options.SeedCanvas!, Scores(layout, prefer), stepsRun: 1, converged: false);
        }

        /// <summary>Read the canvas's question set back out of the prompt it was built with.</summary>
        private static Dictionary<string, StructuredQuestion> QuestionsOnCanvas(StructuredRead read)
        {
            var questions = new Dictionary<string, StructuredQuestion>();
            string[] lines = read.Prompt.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                int colon = lines[i].IndexOf("\": ", StringComparison.Ordinal);
                if (!lines[i].StartsWith('"') || colon < 0) continue;
                string key = lines[i][1..colon];
                string values = lines.Skip(i)
                    .First(l => l.StartsWith("Allowed values: ", StringComparison.Ordinal));
                questions[key] = new StructuredQuestion
                {
                    Instructions = "?",
                    Options = JsonNode.Parse(values["Allowed values: ".Length..])!.AsArray()
                        .Select(n => n!.GetValueKind() switch
                        {
                            JsonValueKind.True => (object?)true,
                            JsonValueKind.False => false,
                            JsonValueKind.Number => n!.GetValue<int>(),
                            _ => n!.GetValue<string>(),
                        }).ToList(),
                };
            }
            return questions;
        }
    }

    private static string JsonText(object? value) => System.Text.Json.JsonSerializer.Serialize(value);

    private static ITokenizer Tokenizer() => new CharTokenizer();

    private static Dictionary<string, StructuredQuestion> TwoQuestions() => new()
    {
        ["refund"] = StructuredQuestion.Boolean("Does the customer ask for a refund?"),
        ["department"] = StructuredQuestion.Choice("Which team should handle this?", "billing", "tech"),
    };

    private static JsonCanvasLayout Compile(IReadOnlyDictionary<string, StructuredQuestion> questions) =>
        JsonCanvasLayout.Compile(Tokenizer(), questions, CanvasLength, eosTokenId: 0);

    // ---- The canvas --------------------------------------------------------

    [Fact]
    public void TheCanvasPinsItsScaffolding_AndLeavesOnlyTheAnswerSlotsFree()
    {
        var layout = Compile(TwoQuestions());

        // Free positions are exactly where candidates of a field disagree; everything else - braces,
        // quoted keys, separators, the padding past the JSON - is held for the whole denoise.
        Assert.NotEmpty(layout.VariablePositions);
        foreach (int pos in layout.VariablePositions)
        {
            Assert.False(layout.PinnedPositions[pos]);
            Assert.NotNull(layout.LogprobTokenIds[pos]);
            Assert.True(layout.LogprobTokenIds[pos].Length > 1);
        }
        for (int pos = 0; pos < layout.CanvasWidth; pos++)
        {
            if (layout.VariablePositions.Contains(pos)) continue;
            Assert.True(layout.PinnedPositions[pos]);
            Assert.Null(layout.LogprobTokenIds[pos]);
        }
        Assert.Equal(CanvasLength, layout.SeedCanvas.Length);
        Assert.Equal(CanvasLength, layout.PinnedPositions.Length);
    }

    [Fact]
    public void TheSeedCanvasDecodesToTheFirstAllowedAnswer()
    {
        var layout = Compile(TwoQuestions());
        var json = JsonNode.Parse(Tokenizer().Decode(layout.SeedCanvas.ToList()))!.AsObject();

        Assert.Equal(new[] { "refund", "department" }, json.Select(x => x.Key));
        Assert.False(json["refund"]!.GetValue<bool>());
        Assert.Equal("billing", json["department"]!.GetValue<string>());
    }

    [Fact]
    public void ShorterCandidatesArePaddedSoAFieldKeepsItsSpan()
    {
        // "tech" is shorter than "billing": both have to occupy the same positions, or the fields after
        // them would sit at different offsets depending on the answer.
        var layout = Compile(TwoQuestions());
        IReadOnlyList<int[]> candidates = layout.CandidateTokens("department");
        Assert.Equal(2, candidates.Count);
        Assert.Equal(candidates[0].Length, candidates[1].Length);
        Assert.EndsWith("   ", new CharTokenizer().Decode(candidates[1].ToList()));
    }

    [Fact]
    public void AQuestionSetTooLargeForTheCanvas_IsRefused()
    {
        var questions = Enumerable.Range(0, 40).ToDictionary(
            i => $"question_number_{i}", i => StructuredQuestion.Boolean("Is it so?"));
        var ex = Assert.Throws<ArgumentException>(() => Compile(questions));
        Assert.Contains("canvas holds", ex.Message);
    }

    [Fact]
    public void ATightCanvasIsOnlyAsWideAsTheAnswersNeed()
    {
        var served = Compile(TwoQuestions());
        var tight = JsonCanvasLayout.Compile(
            Tokenizer(), TwoQuestions(), CanvasLength, eosTokenId: 0, JsonCanvasFit.Tight);

        Assert.Equal(CanvasLength, served.CanvasWidth);
        Assert.True(tight.CanvasWidth < served.CanvasWidth);

        // The JSON is the same; only the padding the forward has to carry is gone - bar one terminator,
        // so the model sees an answer that ended rather than a block that stopped.
        Assert.Equal(
            Tokenizer().Decode(served.SeedCanvas.ToList()),
            Tokenizer().Decode(tight.SeedCanvas.ToList()));
        Assert.Equal(0, tight.SeedCanvas[^1]);
        Assert.True(tight.PinnedPositions[^1]);

        // The answer slots are the same positions either way.
        Assert.Equal(served.VariablePositions, tight.VariablePositions);
    }

    [Fact]
    public void ATightCanvasNeverOutgrowsTheServedOne()
    {
        // The tight width is the JSON plus a terminator, which for a canvas-filling question set would
        // run past the served canvas; it is capped there instead.
        var questions = new Dictionary<string, StructuredQuestion>();
        for (int i = 0; ; i++)
        {
            var trial = new Dictionary<string, StructuredQuestion>(questions)
            {
                [$"q{i}"] = StructuredQuestion.Boolean("Is it so?"),
            };
            try { Compile(trial); }
            catch (ArgumentException) { break; }
            questions = trial;
        }

        var tight = JsonCanvasLayout.Compile(
            Tokenizer(), questions, CanvasLength, eosTokenId: 0, JsonCanvasFit.Tight);
        Assert.True(tight.CanvasWidth <= CanvasLength);
    }

    [Fact]
    public async Task ATightFitAsksTheModelForANarrowerForward()
    {
        var requests = new[] { Request("t", ("urgent", StructuredQuestion.Boolean("Urgent?"))) };

        var (_, wide) = await Predict(requests, _ => true,
            new StructuredPredictOptions { CanvasFit = JsonCanvasFit.ServedCanvas });
        var (predictions, narrow) = await Predict(requests, _ => true,
            new StructuredPredictOptions { CanvasFit = JsonCanvasFit.Tight });

        Assert.Equal(CanvasLength, Assert.Single(wide.Widths));
        Assert.True(Assert.Single(narrow.Widths) < CanvasLength);
        // Same answer either way: the width is what the forward costs, not what it decides.
        Assert.Equal(true, Assert.Single(predictions).Values["urgent"]);
    }

    // ---- The readout -------------------------------------------------------

    /// <summary>Score every allowed token the canvas asks about, with <paramref name="prefer"/>'s tokens
    /// ahead at the positions that belong to it.</summary>
    private static DiffusionPositionLogprobs[] Scores(
        JsonCanvasLayout layout, IReadOnlyDictionary<string, int> prefer)
    {
        int width = layout.CanvasWidth;
        var positions = new DiffusionPositionLogprobs[width];
        var target = new int[width];
        Array.Copy(layout.SeedCanvas, target, width);
        foreach ((string key, int index) in prefer)
        {
            int[] run = layout.CandidateTokens(key)[index];
            Array.Copy(run, 0, target, layout.SlotOffset(key), run.Length);
        }
        for (int pos = 0; pos < width; pos++)
        {
            int[]? ids = layout.LogprobTokenIds[pos];
            positions[pos] = ids is null
                ? new DiffusionPositionLogprobs(Array.Empty<int>(), Array.Empty<float>())
                : new DiffusionPositionLogprobs(
                    ids, ids.Select(id => id == target[pos] ? -0.1f : -3.0f).ToArray());
        }
        return positions;
    }

    [Fact]
    public void TheReadoutPicksTheHighestScoringAllowedAnswer_AndTheJsonIsComplete()
    {
        var layout = Compile(TwoQuestions());
        var prediction = layout.Select(
            "case-1", Scores(layout, new Dictionary<string, int> { ["refund"] = 1, ["department"] = 1 }),
            steps: 1, canvases: 1);

        Assert.Equal(true, prediction.Fields["refund"].Value);
        Assert.Equal("tech", prediction.Fields["department"].Value);

        // The JSON is a member of the allowed language by construction, not by repair.
        var json = JsonNode.Parse(prediction.Json)!.AsObject();
        Assert.Equal(new[] { "refund", "department" }, json.Select(x => x.Key));
        Assert.True(json["refund"]!.GetValue<bool>());
        Assert.Equal("tech", json["department"]!.GetValue<string>());
    }

    [Fact]
    public void TheReadoutSeparatesCandidatesThatShareAPrefix()
    {
        // "approve"/"approve_with_changes"/"reject": the first two agree for seven tokens, so only a later
        // position tells them apart - and the surviving candidate's unique suffix is then forced.
        var questions = new Dictionary<string, StructuredQuestion>
        {
            ["action"] = StructuredQuestion.Choice(
                "What should happen?", "approve", "approve_with_changes", "reject"),
        };
        var layout = Compile(questions);

        foreach (int index in new[] { 0, 1, 2 })
        {
            var prediction = layout.Select(
                "c", Scores(layout, new Dictionary<string, int> { ["action"] = index }), 1, 1);
            Assert.Equal(questions["action"].Options[index], prediction.Fields["action"].Value);
            Assert.Equal(
                questions["action"].Options[index],
                JsonNode.Parse(prediction.Json)!["action"]!.GetValue<string>());
        }
    }

    [Fact]
    public void ScoreQuestionsAnswerWithTheirZeroBasedIndex()
    {
        var questions = new Dictionary<string, StructuredQuestion>
        {
            ["severity"] = StructuredQuestion.Score("How severe?", "low", "medium", "high"),
        };
        var layout = Compile(questions);
        var prediction = layout.Select(
            "c", Scores(layout, new Dictionary<string, int> { ["severity"] = 2 }), 1, 1);

        Assert.Equal(2, prediction.Fields["severity"].Value);
        Assert.Equal(2, JsonNode.Parse(prediction.Json)!["severity"]!.GetValue<int>());
    }

    [Fact]
    public void TheReportedConfidenceIsADistributionOverTheAllowedValues()
    {
        var layout = Compile(TwoQuestions());
        var prediction = layout.Select(
            "c", Scores(layout, new Dictionary<string, int> { ["refund"] = 1 }), 1, 1);

        StructuredFieldAnswer refund = prediction.Fields["refund"];
        Assert.Equal(2, refund.OptionProbabilities.Count);
        Assert.Equal(1.0, refund.OptionProbabilities.Sum(), 5);
        // The chosen value is the one the confidence belongs to, and it leads.
        Assert.Equal(refund.OptionProbabilities[1], refund.Probability, 6);
        Assert.True(refund.Probability > 0.5);
    }

    [Fact]
    public void AReadThatSkippedAPositionTheReadoutNeeds_IsAnError()
    {
        // Silently answering from a missing score would invent a decision; it has to fail loudly.
        var layout = Compile(TwoQuestions());
        var scores = Scores(layout, new Dictionary<string, int>());
        int free = layout.VariablePositions[0];
        scores[free] = new DiffusionPositionLogprobs(Array.Empty<int>(), Array.Empty<float>());

        Assert.Throws<ArgumentException>(() => layout.Select("c", scores, 1, 1));
    }

    // ---- Requests, packing and prediction ----------------------------------

    [Fact]
    public void ARequestIsCheckedBeforeAnythingIsTokenized()
    {
        StructuredRequest With(Dictionary<string, StructuredQuestion> questions) =>
            new() { Id = "c", Document = "text", Questions = questions };

        Assert.Throws<ArgumentException>(() =>
            new StructuredRequest { Id = "c", Document = "", Questions = TwoQuestions() }.Validate());
        Assert.Throws<ArgumentException>(() =>
            With(new Dictionary<string, StructuredQuestion>()).Validate());
        Assert.Throws<ArgumentException>(() => With(new()
        {
            ["a"] = new StructuredQuestion { Instructions = "?", Options = new object?[] { "only" } },
        }).Validate());
        Assert.Throws<ArgumentException>(() => With(new()
        {
            ["a"] = new StructuredQuestion { Instructions = "?", Options = new object?[] { "x", "x" } },
        }).Validate());
        Assert.Throws<ArgumentException>(() => With(new()
        {
            ["a"] = new StructuredQuestion { Instructions = "?", Options = new object?[] { 1.5, 2.5 } },
        }).Validate());

        With(TwoQuestions()).Validate();
    }

    [Fact]
    public void ThePromptCarriesTheDocument_TheQuestionsAndTheirAllowedValues()
    {
        string prompt = StructuredPredictor.BuildPrompt("I was charged twice.",
            new Dictionary<string, StructuredQuestion>
            {
                ["severity"] = StructuredQuestion.Score("How severe?", "low", "high"),
            });

        Assert.Contains("<user_text>\nI was charged twice.\n</user_text>", prompt);
        Assert.Contains("\"severity\": How severe?", prompt);
        Assert.Contains("Criteria: [\"low\",\"high\"]", prompt);
        Assert.Contains("Allowed values: [0,1]", prompt);
    }

    private static StructuredRequest Request(string id, params (string Key, StructuredQuestion Q)[] qs) =>
        new()
        {
            Id = id,
            Document = "Everything is down and we have a demo at noon.",
            Questions = qs.ToDictionary(x => x.Key, x => x.Q),
        };

    /// <summary>Run the predictor with a reader wired to prefer a given answer per key.</summary>
    private static async Task<(IReadOnlyList<StructuredPrediction> Predictions, ScriptedReader Reader)> Predict(
        IReadOnlyList<StructuredRequest> requests,
        Func<string, object?> preferred,
        StructuredPredictOptions? options = null)
    {
        var reader = new ScriptedReader(preferred);
        IReadOnlyList<StructuredPrediction> predictions =
            await new StructuredPredictor(reader).PredictAsync(requests, options).ConfigureAwait(false);
        return (predictions, reader);
    }

    [Fact]
    public async Task APredictionAnswersEveryQuestionOfItsRequest()
    {
        var requests = new[]
        {
            Request("ticket-1",
                ("urgent", StructuredQuestion.Boolean("Does this need a reply within the hour?")),
                ("team", StructuredQuestion.Choice("Who handles it?", "billing", "tech"))),
        };

        var (predictions, reader) = await Predict(
            requests, key => key == "urgent" ? true : "tech");

        StructuredPrediction prediction = Assert.Single(predictions);
        Assert.Equal("ticket-1", prediction.Id);
        Assert.Equal(1, prediction.Canvases);
        Assert.Equal(new[] { "urgent", "team" }, prediction.Fields.Keys);
        Assert.Equal(true, prediction.Values["urgent"]);
        Assert.Equal("tech", prediction.Values["team"]);
        Assert.Equal(1, reader.Reads);      // both questions rode one canvas
    }

    [Fact]
    public async Task QuestionsThatDoNotFitOneCanvas_AreSplitAndMergedBack()
    {
        (string, StructuredQuestion)[] many = Enumerable.Range(0, 12)
            .Select(i => ($"question_number_{i}", StructuredQuestion.Boolean("Is it so?")))
            .ToArray();
        var request = Request("wide", many);

        var predictor = new StructuredPredictor(new ScriptedReader(_ => true));
        Assert.True(predictor.PlanCanvasCounts(new[] { request })[0] > 1);

        var (predictions, _) = await Predict(new[] { request }, _ => true);
        StructuredPrediction prediction = Assert.Single(predictions);

        Assert.True(prediction.Canvases > 1);
        // Split or not, the caller asked one question set and gets one answer, in the order they asked.
        Assert.Equal(many.Select(x => x.Item1), prediction.Fields.Keys);
        Assert.Equal(
            many.Select(x => x.Item1),
            JsonNode.Parse(prediction.Json)!.AsObject().Select(x => x.Key));
        Assert.All(prediction.Values.Values, v => Assert.Equal(true, v));
    }

    [Fact]
    public async Task CanvasesAreBatchedUpToTheRequestedSize()
    {
        StructuredRequest[] requests = Enumerable.Range(0, 7)
            .Select(i => Request($"case-{i}", ("urgent", StructuredQuestion.Boolean("Urgent?"))))
            .ToArray();

        var (predictions, reader) = await Predict(
            requests, _ => true, new StructuredPredictOptions { BatchSize = 3 });

        Assert.Equal(7, predictions.Count);
        Assert.Equal(new[] { 3, 3, 1 }, reader.BatchSizes);
    }

    // ---- The benchmark harness ---------------------------------------------

    private static StructuredBenchmarkCase Case(string id, bool expected) => new()
    {
        Request = Request(id, ("urgent", StructuredQuestion.Boolean("Urgent?"))),
        Expected = new Dictionary<string, object?> { ["urgent"] = expected },
        Workflow = expected ? "urgent-cases" : "calm-cases",
    };

    private static StructuredBenchmark Benchmark(Func<int, bool>? rejectBatch = null) =>
        new(new StructuredPredictor(new ScriptedReader(_ => true, rejectBatch)));

    [Fact]
    public async Task TheBenchmarkScoresAgainstReferences_AndGroupsByWorkflow()
    {
        var cases = new[] { Case("a", true), Case("b", true), Case("c", false) };

        StructuredBenchmarkReport report = await Benchmark().RunAsync(cases,
            new StructuredBenchmarkOptions
            {
                BatchSizes = new[] { 4 },
                Repeats = 2,
                Warmups = 0,
                DeviceUsdPerSecond = 0.001,
                PricingSource = "test",
            });

        StructuredBenchmarkSummary summary = Assert.Single(report.Summaries);
        // The reader always answers true, so the two urgent cases are right and the calm one is wrong.
        Assert.Equal(3, summary.Scored);
        Assert.Equal(2, summary.Correct);
        Assert.Equal(2.0 / 3, summary.Accuracy!.Value, 6);
        Assert.Equal(2, summary.ByWorkflow["urgent-cases"].Correct);
        Assert.Equal(0, summary.ByWorkflow["calm-cases"].Correct);

        // Throughput counts every timed pass; accuracy counts each case once.
        Assert.Equal(6, summary.DocumentsMeasured);
        Assert.Equal(6, summary.JudgmentsMeasured);
        Assert.Equal(0, summary.InconsistentRepeatedDocuments);
        Assert.True(summary.DocumentsPerSecond > 0);
        Assert.NotNull(summary.UsdPer1000Documents);
        Assert.Empty(report.Failures);
        Assert.Equal(3, report.Cases);
        Assert.Equal(3, report.CanvasesPerPass);
        Assert.Equal(0, report.SplitCases);
        // The receipt says what the forward actually ran at, so a throughput number cannot be read
        // without knowing which canvas bought it.
        Assert.Equal(new[] { CanvasLength }, report.CanvasWidths);
        Assert.Equal("ServedCanvas", report.CanvasFit);
    }

    [Fact]
    public async Task TheSweepHalvesPastABatchTheDeviceCannotHold_AndSaysSo()
    {
        // Anything wider than two canvases runs the (pretend) device out of memory.
        StructuredBenchmarkReport report = await Benchmark(rejectBatch: n => n > 2).RunAsync(
            new[] { Case("a", true), Case("b", true), Case("c", true), Case("d", true) },
            new StructuredBenchmarkOptions { BatchSizes = new[] { 8, 4, 2, 1 }, Repeats = 1, Warmups = 0 });

        Assert.Equal(new[] { 8, 4 }, report.Failures.Select(f => f.RequestedBatchSize));
        // Reported is the largest batch that actually held, not one chosen in advance.
        Assert.Equal(2, Assert.Single(report.Summaries).RequestedBatchSize);
    }

    [Fact]
    public async Task ARealFailureIsNotMistakenForABatchThatWasTooBig()
    {
        var benchmark = new StructuredBenchmark(new StructuredPredictor(
            new ScriptedReader(_ => throw new InvalidOperationException("the schema is wrong"))));

        await Assert.ThrowsAsync<InvalidOperationException>(() => benchmark.RunAsync(
            new[] { Case("a", true) },
            new StructuredBenchmarkOptions { BatchSizes = new[] { 2, 1 }, Repeats = 1, Warmups = 0 }));
    }

    [Fact]
    public async Task TheReceiptSerializesWithItsCaveats()
    {
        StructuredBenchmarkReport report = await Benchmark().RunAsync(
            new[] { Case("a", true) },
            new StructuredBenchmarkOptions { BatchSizes = new[] { 2 }, Repeats = 1, Warmups = 0 });

        var json = JsonNode.Parse(report.ToJson())!.AsObject();
        Assert.Equal(1, json["cases"]!.GetValue<int>());
        Assert.Equal(1, json["steps"]!.GetValue<int>());
        // A number without its scope is a number someone will quote out of context.
        Assert.Contains("Not an invoice", json["cost_scope"]!.GetValue<string>());
        Assert.Contains("not isolated", json["scope"]!.GetValue<string>());
        Assert.NotNull(json["summaries"]![0]!["documents_per_second"]);
    }
}
