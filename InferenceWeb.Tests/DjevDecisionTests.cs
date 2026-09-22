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
// Parity of TensorSharp.Structured.Decisions with djev, the reference implementation, without a model.
//
// Fixtures/Djev/djev_fixtures.json was produced by djev's own engine (generate_djev_fixtures.py drives
// djev/engine.py against a mock vLLM transport). Each case records every read djev made - system prompt,
// user turn, seeded canvas, width, label ids - and the response it built. The same cases are replayed here
// through DiffusionAgent with the same tokenizer and the same scores, and must reproduce all of it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured.Decisions;
using Xunit;
using JsonValue = System.Text.Json.Nodes.JsonValue;

namespace InferenceWeb.Tests;

public class DjevDecisionTests
{
    // ---- The fixture's tokenizer and scores, duplicated from generate_djev_fixtures.py ----------------

    internal sealed class WordTokenizer : ITokenizer
    {
        private static readonly Regex Pieces = new(@"<\|channel>|<channel\|>|[A-Za-z]+|\d|\s|.", RegexOptions.Singleline);

        public const int Vocabulary = 262144;
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 2;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => Vocabulary;

        public List<int> Encode(string text, bool addSpecial = true) =>
            Pieces.Matches(text).Select(m => Id(m.Value)).ToList();

        public static int Id(string piece)
        {
            uint h = 2166136261;
            foreach (byte b in Encoding.UTF8.GetBytes(piece)) h = (h ^ b) * 16777619;
            return 300 + (int)(h % (Vocabulary - 300));
        }

        public string Decode(List<int> ids) => string.Join(" ", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => Id(tokenStr);
    }

    internal static double FixtureScore(IReadOnlyList<int> canvas, int position, int tokenId)
    {
        ulong h = 1469598103934665603UL;
        foreach (long value in canvas.Select(c => (long)c).Append(position).Append(tokenId))
            h = (h ^ (uint)value) * 1099511628211UL;
        h ^= h >> 29;
        return -((h % 4096) / 512.0) - 0.125;
    }

    /// <summary>Stands in for patched vLLM: records what it is asked and scores the requested ids.</summary>
    internal sealed class RecordingReader : IDecisionReader
    {
        private readonly Func<IReadOnlyList<int>, int, int, double> _score;

        public RecordingReader(Func<IReadOnlyList<int>, int, int, double>? score = null) => _score = score ?? FixtureScore;

        public List<(string System, string User)> Prompts { get; } = new();
        public List<DecisionRead> Reads { get; } = new();
        public List<int> BatchSizes { get; } = new();
        public int PromptLength { get; set; } = 20;

        public ITokenizer Tokenizer { get; } = new WordTokenizer();
        public int CanvasLength => 256;
        public int VocabSize => WordTokenizer.Vocabulary;
        public int MaxContextLength { get; set; } = 32768;
        public string ModelName => "fixture";

        public int[] EncodeChat(string system, string user)
        {
            Prompts.Add((system, user));
            return Enumerable.Repeat(7, PromptLength).ToArray();
        }

        public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(IReadOnlyList<DecisionRead> reads, CancellationToken cancellationToken = default)
        {
            BatchSizes.Add(reads.Count);
            Reads.AddRange(reads);
            var results = reads.Select(read =>
            {
                int[] canvas = read.Options.SeedCanvas;
                var rows = new DiffusionPositionLogprobs[canvas.Length];
                for (int pos = 0; pos < canvas.Length; pos++)
                {
                    int[] ids = read.Options.LogprobTokenIds[pos] ?? Array.Empty<int>();
                    rows[pos] = new DiffusionPositionLogprobs(ids, ids.Select(id => (float)_score(canvas, pos, id)).ToArray());
                }
                return new DiffusionReadResult(canvas, rows, stepsRun: 1, converged: false);
            }).ToList();
            return Task.FromResult<IReadOnlyList<DiffusionReadResult>>(results);
        }
    }

    private static JsonObject Fixtures()
    {
        using Stream stream = typeof(DjevDecisionTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Djev.djev_fixtures.json")!;
        return JsonNode.Parse(stream)!.AsObject();
    }

    public static IEnumerable<object[]> CaseNames() =>
        Fixtures()["cases"]!.AsArray().Select(c => new object[] { c!["name"]!.GetValue<string>() });

    // ---- Parity with djev's engine ------------------------------------------------------------------

    [Theory]
    [MemberData(nameof(CaseNames))]
    public async Task ReplaysDjevEngineExactly(string name)
    {
        JsonNode fixture = Fixtures()["cases"]!.AsArray().First(c => c!["name"]!.GetValue<string>() == name)!;
        var reader = new RecordingReader();
        using var agent = new DiffusionAgent(reader, new DiffusionAgentOptions
        {
            Canvas = new DecisionCanvasOptions { Canvas = 128, Compact = fixture["compact"]!.GetValue<bool>() },
            BatchSize = 64,
        });

        DecisionRequest request = DecisionRequest.FromJson(fixture["request"]!);
        DecisionResult result = await agent.PredictAsync(request);

        JsonArray expectedReads = fixture["reads"]!.AsArray();
        Assert.Equal(expectedReads.Count, reader.Reads.Count);
        for (int i = 0; i < expectedReads.Count; i++)
        {
            JsonNode expected = expectedReads[i]!;
            DecisionRead actual = reader.Reads[i];
            Assert.Equal(expected["width"]!.GetValue<int>(), actual.Options.CanvasWidth);
            Assert.Equal(expected["canvas"]!.AsArray().Select(t => t!.GetValue<int>()), actual.Options.SeedCanvas);
            Assert.Equal(1, actual.Options.MaxSteps);
            Assert.True(actual.Options.ReadOnly);
            Assert.Null(actual.Options.PinnedPositions);
            // djev asks for the union of every label id; each slot here asks for its own labels, all of them in it.
            var requested = actual.Options.LogprobTokenIds.Where(r => r != null).SelectMany(r => r).Distinct().Order();
            Assert.Equal(expected["label_ids"]!.AsArray().Select(t => t!.GetValue<int>()), requested);
        }

        // Prompts are compiled once per distinct schema; each must be one djev sent.
        var sent = expectedReads.Select(r => (r!["system"]!.GetValue<string>(), r["user"]!.GetValue<string>())).ToHashSet();
        Assert.All(reader.Prompts, p => Assert.Contains(p, sent));
        Assert.Equal(sent.Count, reader.Prompts.Distinct().Count());

        Assert.Equal(expectedReads.Sum(r => r!["max_tokens"]!.GetValue<int>()), result.Usage.OutputTokens);
        Assert.Equal(20 * expectedReads.Count, result.Usage.InputTokens);

        JsonObject body = result.ToJson();
        JsonNode reference = fixture["response"]!;
        AssertJsonEqual(reference["answers"], body["answers"], "answers");
        if (reference["diagnostics"] is JsonObject diagnostics)
        {
            diagnostics.Remove("engine");
            var ours = body["diagnostics"]!.AsObject();
            ours.Remove("engine");
            AssertJsonEqual(diagnostics, ours, "diagnostics");
        }
        else
        {
            Assert.Null(body["diagnostics"]);
        }
    }

    private static void AssertJsonEqual(JsonNode? expected, JsonNode? actual, string path)
    {
        switch (expected)
        {
            case null:
                Assert.True(actual is null, $"{path}: expected null, got {actual?.ToJsonString()}");
                return;
            case JsonObject e:
            {
                var a = Assert.IsType<JsonObject>(actual);
                Assert.Equal(e.Select(p => p.Key), a.Select(p => p.Key));
                foreach ((string key, JsonNode? value) in e) AssertJsonEqual(value, a[key], $"{path}.{key}");
                return;
            }
            case JsonArray e:
            {
                var a = Assert.IsType<JsonArray>(actual);
                Assert.Equal(e.Count, a.Count);
                for (int i = 0; i < e.Count; i++) AssertJsonEqual(e[i], a[i], $"{path}[{i}]");
                return;
            }
            default:
                Assert.NotNull(actual);
                if (expected.GetValueKind() == JsonValueKind.Number)
                {
                    double x = double.Parse(expected.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture);
                    double y = double.Parse(actual!.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture);
                    Assert.True(Math.Abs(x - y) <= 1e-12 * Math.Max(1, Math.Abs(x)), $"{path}: {x} != {y}");
                }
                else
                {
                    Assert.Equal(expected.ToJsonString(), actual!.ToJsonString());
                }
                return;
        }
    }

    // ---- Bit-exact pieces ------------------------------------------------------------------------------

    [Theory]
    [InlineData("djev-canvas-v1:0", 262144, new long[] { 129865, 199226, 182241, 48422, 108401, 77702 })]
    [InlineData("djev-canvas-v1:17", 262144, new long[] { 247537, 131297, 109763, 6606, 115393, 96262 })]
    [InlineData("djev-canvas-v1:-3", 262144, new long[] { 113040, 111153, 80703, 47772, 175697, 134287 })]
    [InlineData("djev-canvas-v1:7919", 1000, new long[] { 604, 950, 969, 613 })]
    public void PythonRandomMatchesCPython(string seed, long n, long[] expected)
    {
        var rng = new PythonRandom(seed);
        Assert.Equal(expected, expected.Select(_ => rng.RandRange(n)).ToArray());
    }

    [Fact]
    public void IndependentSeedMatchesDjev()
    {
        BigInteger seed = DiffusionAgent.IndependentSeed(BigInteger.Zero, "abc", 1);
        Assert.Equal(BigInteger.Parse("74676927930537205717939509085520783704123641664529789116584016694666257950137"), seed);
        var rng = new PythonRandom("djev-canvas-v1:" + seed);
        Assert.Equal(new long[] { 225572, 106168, 251349 }, new[] { rng.RandRange(262144), rng.RandRange(262144), rng.RandRange(262144) });
    }

    [Theory]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(1e16, "1e+16")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(0.00001, "1e-05")]
    [InlineData(1.5, "1.5")]
    [InlineData(-2.0, "-2.0")]
    [InlineData(123456789.123, "123456789.123")]
    [InlineData(1e-7, "1e-07")]
    [InlineData(1.2345e22, "1.2345e+22")]
    [InlineData(0.0, "0.0")]
    public void FloatsAreSpelledAsPythonReprs(double value, string expected) =>
        Assert.Equal(expected, PythonJson.FormatFloat(value));

    [Fact]
    public void JsonIsSpelledAsCompactPythonDumps()
    {
        JsonNode node = JsonNode.Parse("{\"a\":\"é\\u2028\\u007f\\u0001\",\"b\":[1,2.5,2.0,1e16],\"c\":null,\"d\":true}")!;
        string raw = "\"é" + (char)0x2028 + (char)0x7f + "\\u0001\"";   // U+2028 and DEL pass through; controls escape
        Assert.Equal("{\"a\":" + raw + ",\"b\":[1,2.5,2.0,1e+16],\"c\":null,\"d\":true}", PythonJson.Dumps(node));
        Assert.Equal("{\"b\":1,\"a\":2}", PythonJson.Dumps(JsonNode.Parse("{\"b\":1,\"a\":2}")));
        Assert.Equal("{\"a\":2,\"b\":1}", PythonJson.Dumps(JsonNode.Parse("{\"b\":1,\"a\":2}"), sortKeys: true));
    }

    [Fact]
    public void FsumIsExactlyRounded()
    {
        Assert.Equal(1.0, DecisionContracts.Fsum(Enumerable.Repeat(0.1, 10).ToArray()));
        Assert.Equal(0.0, DecisionContracts.Fsum(new[] { 1e100, 1.0, -1e100 }) - 1.0);
    }

    // ---- The compiled read -------------------------------------------------------------------------------

    private static DiffusionAgent Agent(RecordingReader reader, int canvas = 128) =>
        new(reader, new DiffusionAgentOptions { Canvas = new DecisionCanvasOptions { Canvas = canvas } });

    [Fact]
    public void CompilesDjevTemplateAndSlots()
    {
        using var agent = Agent(new RecordingReader());
        CompiledDecisionSchema schema = agent.Compile(Presets.Triage());

        var tokenizer = new WordTokenizer();
        var expected = tokenizer.Encode(DecisionSchemaCompiler.Scaffold)
            .Concat(tokenizer.Encode("0:A\n1:no\n2:0\n3:no\n4:no")).ToList();
        Assert.Equal(expected, schema.Template);
        Assert.Equal(32, schema.CanvasWidth);
        Assert.Equal(5, schema.Slots.Count);
        Assert.Equal(new[] { "A", "B", "C", "D", "E", "F" }.Select(WordTokenizer.Id), schema.Slots[0].TokenIds);
        Assert.Equal(new[] { WordTokenizer.Id("no"), WordTokenizer.Id("yes") }, schema.Slots[1].TokenIds);
        Assert.Equal(Enumerable.Range(0, 4).Select(i => WordTokenizer.Id(i.ToString())), schema.Slots[2].TokenIds);
        Assert.StartsWith(DecisionSchemaCompiler.Preamble, schema.SystemPrompt);
        Assert.Contains("  A: refund — money returned or a duplicate charge reversed", schema.SystemPrompt);
        Assert.EndsWith("\nReply with one line per question, in order: \"id:label\". Do not add explanations.", schema.SystemPrompt);
        Assert.DoesNotContain("refund_requested", schema.SystemPrompt);   // question ids never reach the model
        Assert.Same(schema, agent.Compile(Presets.Triage()));             // the schema cache holds schemas, not answers
    }

    [Fact]
    public void SeedCanvasHoldsTemplateTerminatorPadAndNoise()
    {
        using var agent = Agent(new RecordingReader());
        CompiledDecisionSchema schema = agent.Compile(new QuestionSet().Add("q", Question.Noul("Urgent?")));
        int[] canvas = DecisionSchemaCompiler.SeedCanvas(schema, 7, WordTokenizer.Vocabulary);
        Assert.Equal(schema.CanvasWidth, canvas.Length);
        Assert.Equal(0, canvas.Length % 16);
        int slot = schema.Slots[0].Position;
        for (int i = 0; i < schema.Template.Count; i++)
            if (i != slot) Assert.Equal(schema.Template[i], canvas[i]);
        Assert.Equal(106, canvas[schema.Template.Count]);
        Assert.All(canvas.Skip(schema.Template.Count + 1), t => Assert.Equal(0, t));
        Assert.Equal(new PythonRandom("djev-canvas-v1:7").RandRange(WordTokenizer.Vocabulary), canvas[slot]);
        Assert.Equal(canvas, DecisionSchemaCompiler.SeedCanvas(schema, 7, WordTokenizer.Vocabulary));
    }

    [Fact]
    public void TemplateThatDoesNotFitTheCanvasFailsBeforeAnyRead()
    {
        var reader = new RecordingReader();
        using var agent = Agent(reader, canvas: 8);
        Assert.Throws<DecisionSchemaException>(() => agent.SystemOne("Cancel my plan", Presets.Triage()));
        Assert.Empty(reader.Reads);
    }

    [Fact]
    public void LabelsThatAreNotOneTokenInContextAreRefused()
    {
        // djev's test_context_sensitive_token_length_is_rejected: "yes" costs an extra token in context.
        using var agent = new DiffusionAgent(new OddTokenizerReader());
        var ex = Assert.Throws<DecisionSchemaException>(() =>
            agent.SystemOne("x", new QuestionSet().Add("q", Question.Noul("Urgent?"))));
        Assert.Contains("single token", ex.Message);
    }

    /// <summary>A tokenizer that splits "yes" into two tokens, as a context-sensitive BPE could.</summary>
    private sealed class OddTokenizerReader : IDecisionReader
    {
        private sealed class Odd : ITokenizer
        {
            private readonly WordTokenizer _inner = new();
            public string[] Vocab => Array.Empty<string>();
            public int BosTokenId => 2;
            public int[] EosTokenIds => new[] { 1 };
            public int VocabSize => WordTokenizer.Vocabulary;
            public List<int> Encode(string text, bool addSpecial = true)
            {
                var ids = _inner.Encode(text);
                if (text.Contains("yes")) ids.Add(700);
                return ids;
            }
            public string Decode(List<int> ids) => "";
            public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
        public ITokenizer Tokenizer { get; } = new Odd();
        public int CanvasLength => 256;
        public int VocabSize => WordTokenizer.Vocabulary;
        public int MaxContextLength => 32768;
        public string ModelName => "odd";
        public int[] EncodeChat(string system, string user) => new int[20];
        public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(IReadOnlyList<DecisionRead> reads, CancellationToken ct = default) =>
            throw new InvalidOperationException("no read should be attempted");
    }

    [Fact]
    public void ExactLabelLogprobsMakeTheNoulAndItsMass()
    {
        // djev's test_exact_logprobs_produce_noul_and_mass_without_leaking_ids: no = log .1, yes = log .3.
        int no = WordTokenizer.Id("no"), yes = WordTokenizer.Id("yes");
        var reader = new RecordingReader((_, _, id) => id == no ? Math.Log(.1) : id == yes ? Math.Log(.3) : -20);
        using var agent = Agent(reader);
        DecisionResult result = agent.SystemOne(new JsonObject { ["text"] = "Cancel my plan" },
            new QuestionSet().Add("hidden customer name", Question.Noul("Wants cancellation?")),
            new DecisionOptions { Diagnostics = true, Seed = 7 });
        Assert.Equal(0.75, result["hidden customer name"].Noul!.Value, 6);
        Assert.Equal(0.4, result.Diagnostics!["questions"]!["hidden customer name"]!["label_mass"]!.GetValue<double>(), 6);
        Assert.Equal("unvalidated", result.Diagnostics["calibration"]!.GetValue<string>());
        Assert.DoesNotContain(reader.Prompts, p => p.System.Contains("hidden customer name") || p.User.Contains("hidden customer name"));
        Assert.Equal("{\"text\":\"Cancel my plan\"}", reader.Prompts.Single().User);
    }

    [Fact]
    public void MissingOrInvalidEvidenceFailsInsteadOfFabricatingAnAnswer()
    {
        var set = new QuestionSet().Add("q", Question.Noul("Urgent?"));
        foreach (Func<IReadOnlyList<int>, int, int, double> score in new Func<IReadOnlyList<int>, int, int, double>[]
                 {
                     (_, _, _) => double.NaN,
                     (_, _, _) => 0.5,                        // a log probability above zero
                     (_, _, _) => double.NegativeInfinity,    // no finite evidence at all
                     (_, _, _) => -10000,                     // vLLM's -inf sentinel, everywhere
                 })
        {
            using var agent = Agent(new RecordingReader(score));
            Assert.Throws<DecisionBackendException>(() => agent.SystemOne("x", set));
        }
    }

    [Fact]
    public void ImpossibleLabelsKeepZeroMass()
    {
        int yes = WordTokenizer.Id("yes");
        using var agent = Agent(new RecordingReader((_, _, id) => id == yes ? -9999 : -1));
        Assert.Equal(0.0, agent.SystemOne("x", new QuestionSet().Add("q", Question.Noul("Urgent?")))["q"].Noul);
    }

    [Fact]
    public void FixedSeedRepeatsTheCanvasAndNeverCachesTheAnswer()
    {
        var reader = new RecordingReader();
        using var agent = Agent(reader);
        var set = new QuestionSet().Add("q", Question.Noul("Urgent?"));
        agent.SystemOne("x", set, new DecisionOptions { Seed = 17 });
        agent.SystemOne("x", set, new DecisionOptions { Seed = 17 });
        agent.SystemOne("x", set, new DecisionOptions { Seed = null });
        Assert.Equal(3, reader.Reads.Count);
        Assert.Equal(reader.Reads[0].Options.SeedCanvas, reader.Reads[1].Options.SeedCanvas);
    }

    [Fact]
    public void RequestsBatchTogetherButAnswerAsTheyWouldAlone()
    {
        var alone = new RecordingReader();
        using var single = Agent(alone);
        var requests = new[] { "I was charged twice", "The app crashes", "Where is my order?" }
            .Select(s => DecisionRequest.Create(s, Presets.Triage(), new DecisionOptions { Samples = 2 })).ToList();
        var expected = requests.Select(r => single.PredictAsync(r).GetAwaiter().GetResult().ToJsonString()).ToList();

        var batched = new RecordingReader();
        using var agent = new DiffusionAgent(batched, new DiffusionAgentOptions { BatchSize = 4 });
        var results = agent.PredictAsync(requests).GetAwaiter().GetResult();
        Assert.Equal(expected, results.Select(r => r.ToJsonString()));
        Assert.Equal(new[] { 4, 2 }, batched.BatchSizes);
    }

    [Fact]
    public void ContextLimitIsCheckedBeforeAnyRead()
    {
        var reader = new RecordingReader { MaxContextLength = 40, PromptLength = 30 };
        using var agent = Agent(reader);
        var ex = Assert.Throws<DecisionSchemaException>(() => agent.SystemOne("x", new QuestionSet().Add("q", Question.Noul("?"))));
        Assert.Contains("context", ex.Message);
        Assert.Empty(reader.Reads);
    }

    // ---- The request contract --------------------------------------------------------------------------

    [Theory]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\"}},\"options\":{\"samples\":5}}", "samples")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\"}},\"options\":{\"steps\":2}}", "steps")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\"}},\"options\":{\"score_mode\":\"independent_levels\"}}", "independent")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[\"only\"]}}}", "levels")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{}}}}", "options")]
    [InlineData("{\"state\":\"x\",\"questions\":{}}", "questions")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\"}},\"images\":[\"data:image/png;base64,AA==\"]}", "image")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{\"a\":{\"image\":\"data:image/png;base64,AA==\"}}}}}", "image")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"noul\",\"extra\":1}}}", "extra")]
    [InlineData("{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"maybe\"}}}", "maybe")]
    [InlineData("{\"state\":7,\"questions\":{\"q\":{\"type\":\"noul\"}}}", "state")]
    public void RefusesWhatDjevRefuses(string body, string reason)
    {
        using var agent = Agent(new RecordingReader());
        var ex = Assert.ThrowsAny<DecisionSchemaException>(() =>
            agent.PredictAsync(DecisionRequest.FromJson(body)).GetAwaiter().GetResult());
        Assert.Contains(reason, ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LimitsCountCodePointsAndCompactSortedJson()
    {
        var state = new JsonObject { ["text"] = new string('x', DecisionContracts.MaxStateCharacters - 11) }; // {"text":"…"} adds 11
        Assert.Equal(DecisionContracts.MaxStateCharacters, DecisionContracts.CharacterCount(state));
        using var agent = Agent(new RecordingReader());
        var set = new QuestionSet().Add("q", Question.Noul("?"));
        agent.SystemOne(state, set);
        state["text"] = state["text"]!.GetValue<string>() + "x";
        Assert.Throws<DecisionSchemaException>(() => agent.SystemOne(state, set));
        Assert.Equal(1, DecisionContracts.CharacterCount(JsonValue.Create("😀")));
    }

    [Fact]
    public void AnswersFollowDjevsArithmetic()
    {
        Answer score = DecisionContracts.AnswerFrom(Question.Score("?", "a", "b", "c"), new[] { 0.2, 0.5, 0.3 });
        Assert.Equal(1.1, score.Score!.Value, 12);
        Assert.Equal("1", score.Label);
        Answer tie = DecisionContracts.AnswerFrom(Question.Choice("?", "x", "y"), new[] { 0.5, 0.5 });
        Assert.Equal("x", tie.Choice);
        Assert.Equal(0.0, tie.Confidence!.Value, 12);
        Answer single = DecisionContracts.AnswerFrom(Question.Choice("?", "only"), new[] { 1.0 });
        Assert.Equal(1.0, single.Confidence);
        Assert.Throws<DecisionBackendException>(() => DecisionContracts.AnswerFrom(Question.Noul("?"), new[] { 0.5, 0.6 }));
    }

    [Fact]
    public void QuestionsRoundTripDjevsWireFormat()
    {
        string json = "{\"type\":\"noul\",\"instructions\":\"Permitted?\",\"criteria\":{\"true\":\"yes it is\",\"false\":null}}";
        Question q = Question.FromJson(JsonNode.Parse(json)!);
        Assert.Equal(json, PythonJson.Dumps(q.ToJson()));
        Assert.Equal("{\"type\":\"noul\",\"instructions\":\"?\",\"criteria\":null}", PythonJson.Dumps(Question.Noul("?").ToJson()));
        Assert.Equal("{\"type\":\"score\",\"instructions\":\"?\",\"criteria\":[\"a\",\"b\"]}",
            PythonJson.Dumps(Question.Score("?", "a", "b").ToJson()));
    }
}
