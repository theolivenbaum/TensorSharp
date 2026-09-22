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
// Typed DiffusionGemma decisions from the command line - the Laya CLI's predict / bench, plus a JevBench replay.
//
//   predict   answer a question set (a preset, --question flags, or a djev request body) about some state
//   bench     time one question set: warm-up, N timed runs, median ms and ms per question
//   jevbench  replay jevbench's public decisions and write a scored receipt
//   plan      what a JevBench run would cost - reads, canvas widths, prompt tokens - without reading
//
// --model <gguf> loads a checkpoint; --mock runs the same pipeline over a stand-in reader with no model, which
// proves the harness (compilation, batching, scoring, the receipt) and nothing about the model.
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using DiffusionDecisionBench;
using TensorSharp.Runtime;
using TensorSharp.Structured.Decisions;
using TensorSharp.Structured.Evals;

var options = CommandLine.Parse(args);
string command = options.Positional.FirstOrDefault() ?? "help";
try
{
    return command switch
    {
        "predict" => await Predict(options),
        "bench" => await Bench(options),
        "jevbench" => await JevBench(options),
        "plan" => Plan(options),
        "probe" => await Probe(options),
        "presets" => ListPresets(),
        _ => Help(),
    };
}
catch (Exception ex) when (ex is ArgumentException or FileNotFoundException or DirectoryNotFoundException
                           or DecisionBackendException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 2;
}

static int Help()
{
    Console.WriteLine("""
        DiffusionDecisionBench - typed decisions on DiffusionGemma, the way djev reads them

        usage: DiffusionDecisionBench <command> [options]

          predict    --preset triage --text "I was charged twice"
                     --question 'urgent=noul:Is this time critical?'
                     --question 'team=choice:Who handles this?|billing,support,security'
                     --question 'impact=score:How bad is it?|none,minor,major,outage'
                     --request request.json            (a djev POST /v1/request body)
          bench      same inputs as predict, plus --iterations 5
          jevbench   --jevbench <checkout> [--splits easy,original,hard] [--limit N] [--out receipt.json]
                     [--pacing serial|batched] [--price-per-m 0.035] [--device-usd-per-hour X] [--endpoint gpu]
          plan       --jevbench <checkout> [--splits ...]   reads, canvas widths and prompt tokens, no reads run
          probe      same inputs as predict: the model's top tokens at every canvas position of a joint read
          presets    list the built-in question sets

        model:       --model <file.gguf> [--backend ggmlcuda|ggmlmetal|ggmlcpu|cuda|cpu|mlx]  or  --mock
        state:       --text "..." | --state-json '{...}' | --state-file ticket.json
        read:        [--samples 1-4] [--seed N|random] [--isolation joint|independent]
                     [--score-mode categorical|independent_levels] [--diagnostics]
        engine:      [--canvas 128] [--no-compact] [--batch-size 16] [--max-model-len N]
        """);
    return 0;
}

static int ListPresets()
{
    foreach (string name in Presets.Names)
    {
        QuestionSet set = Presets.ByName(name);
        Console.WriteLine($"{name}: {string.Join(", ", set.Ids)}");
    }
    return 0;
}

static DiffusionAgent OpenAgent(CommandLine options)
{
    var agentOptions = new DiffusionAgentOptions
    {
        Canvas = new DecisionCanvasOptions
        {
            Canvas = options.Int("canvas") ?? 128,
            Compact = !options.Has("no-compact"),
        },
        BatchSize = options.Int("batch-size") ?? 16,
        MaxModelLength = options.Int("max-model-len"),
    };
    if (options.Has("mock"))
        return new DiffusionAgent(new MockDecisionReader(), agentOptions);

    string model = options.Value("model")
        ?? Environment.GetEnvironmentVariable("TS_DIFFUSION_MODEL")
        ?? throw new ArgumentException("provide --model <file.gguf> (or TS_DIFFUSION_MODEL), or --mock");
    if (!File.Exists(model)) throw new FileNotFoundException($"model not found: {model}", model);
    BackendType backend = ParseBackend(options.Value("backend") ?? (OperatingSystem.IsMacOS() ? "ggmlmetal" : "ggmlcpu"));
    var watch = Stopwatch.StartNew();
    Console.Error.WriteLine($"loading {Path.GetFileName(model)} on {backend} ...");
    DiffusionAgent agent = DiffusionAgent.Load(model, backend, agentOptions);
    Console.Error.WriteLine($"loaded in {watch.Elapsed.TotalSeconds:F1} s");
    return agent;
}

static BackendType ParseBackend(string name) => name.Replace("_", "").ToLowerInvariant() switch
{
    "ggmlcuda" => BackendType.GgmlCuda,
    "ggmlmetal" => BackendType.GgmlMetal,
    "ggmlcpu" => BackendType.GgmlCpu,
    "ggmlvulkan" => BackendType.GgmlVulkan,
    "cuda" => BackendType.Cuda,
    "cpu" => BackendType.Cpu,
    "mlx" => BackendType.Mlx,
    _ => throw new ArgumentException($"unknown backend '{name}'"),
};

static DecisionOptions ReadOptions(CommandLine options) => new()
{
    Samples = options.Int("samples") ?? 1,
    Seed = options.Value("seed") switch
    {
        null => BigInteger.Zero,
        "random" or "null" => null,
        string s => BigInteger.Parse(s, CultureInfo.InvariantCulture),
    },
    Isolation = options.Value("isolation") == "independent" ? DecisionIsolation.Independent : DecisionIsolation.Joint,
    ScoreMode = options.Value("score-mode") == "independent_levels" ? ScoreMode.IndependentLevels : ScoreMode.Categorical,
    Diagnostics = options.Has("diagnostics"),
};

static DecisionRequest ReadRequest(CommandLine options)
{
    if (options.Value("request") is string path)
    {
        DecisionRequest fromFile = DecisionRequest.FromJson(File.ReadAllText(path));
        return options.Has("samples") || options.Has("seed") || options.Has("isolation") || options.Has("diagnostics")
            ? fromFile with { Options = ReadOptions(options) }
            : fromFile;
    }

    JsonNode state;
    if (options.Value("state-file") is string file)
    {
        string content = File.ReadAllText(file);
        state = file.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? JsonNode.Parse(content)! : JsonValue.Create(content);
    }
    else if (options.Value("state-json") is string json) state = JsonNode.Parse(json)!;
    else state = JsonValue.Create(options.Value("text") ?? throw new ArgumentException("provide --text, --state-json, --state-file or --request"));

    QuestionSet questions = options.Values("question") is { Count: > 0 } specs
        ? ParseQuestions(specs)
        : Presets.ByName(options.Value("preset") ?? "triage");
    return new DecisionRequest { State = state, Questions = questions, Options = ReadOptions(options) };
}

// id=type:instructions|opt1,opt2   (choice options and score levels after the bar)
static QuestionSet ParseQuestions(IReadOnlyList<string> specs)
{
    var set = new QuestionSet();
    foreach (string spec in specs)
    {
        int eq = spec.IndexOf('='), colon = spec.IndexOf(':');
        if (eq < 1 || colon < eq) throw new ArgumentException($"--question '{spec}': expected id=type:instructions[|options]");
        string id = spec[..eq], type = spec[(eq + 1)..colon], rest = spec[(colon + 1)..];
        int bar = rest.LastIndexOf('|');
        string instructions = bar < 0 ? rest : rest[..bar];
        string[] items = bar < 0 ? Array.Empty<string>() : rest[(bar + 1)..].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        set.Add(id, type switch
        {
            "noul" => Question.Noul(instructions),
            "choice" => Question.Choice(instructions, items),
            "score" => Question.Score(instructions, items.Cast<object?>().ToArray()),
            _ => throw new ArgumentException($"--question '{spec}': type must be noul, choice or score"),
        });
    }
    return set;
}

static async Task<int> Predict(CommandLine options)
{
    DecisionRequest request = ReadRequest(options);
    using DiffusionAgent agent = OpenAgent(options);
    DecisionResult result = await agent.PredictAsync(request);
    Console.WriteLine(result.ToJsonString(indented: true));
    Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "compile {0:F1} ms, model {1:F1} ms, canvas {2}, {3} input tokens",
        result.CompileMs, result.ModelMs, string.Join("/", result.CanvasWidths), result.Usage.InputTokens));
    return 0;
}

static async Task<int> Bench(CommandLine options)
{
    DecisionRequest request = ReadRequest(options);
    int iterations = options.Int("iterations") ?? 5;
    using DiffusionAgent agent = OpenAgent(options);

    await agent.PredictAsync(request);   // warm the kernels, the allocator and the schema cache
    var timings = new List<double>();
    for (int i = 0; i < iterations; i++)
    {
        var watch = Stopwatch.StartNew();
        DecisionResult result = await agent.PredictAsync(request);
        watch.Stop();
        timings.Add(watch.Elapsed.TotalMilliseconds);
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "run {0}: {1:F1} ms for {2} questions / {3} input tokens (compile {4:F1} ms, model {5:F1} ms, canvas {6})",
            i + 1, timings[^1], request.Questions.Count, result.Usage.InputTokens, result.CompileMs, result.ModelMs,
            string.Join("/", result.CanvasWidths)));
    }
    timings.Sort();
    double median = timings[timings.Count / 2];
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "median {0:F1} ms, {1:F1} ms/question ({2}, {3})",
        median, median / Math.Max(1, request.Questions.Count), agent.ModelName,
        request.Options.Isolation == DecisionIsolation.Joint ? "joint" : "independent"));
    return 0;
}

// What the model puts at each canvas position of the joint read, top-K at temperature 1: the check for a read
// whose label mass is low - is the slot predicting a label at all, or something else entirely?
static async Task<int> Probe(CommandLine options)
{
    DecisionRequest request = ReadRequest(options);
    using DiffusionAgent agent = OpenAgent(options);
    CompiledDecisionSchema schema = agent.Compile(request.Questions);
    int[] prompt = agent.Reader.EncodeChat(schema.SystemPrompt, DecisionSchemaCompiler.Describe(request.State));
    int[] canvas = DecisionSchemaCompiler.SeedCanvas(schema, request.Options.Seed ?? 0, agent.Reader.VocabSize);
    int k = options.Int("top") ?? 5;
    var read = new DecisionRead
    {
        PromptTokens = prompt,
        Options = new TensorSharp.Models.DiffusionReadOptions
        {
            ReadOnly = true, MaxSteps = 1, CanvasWidth = schema.CanvasWidth, SeedCanvas = canvas, TopLogprobs = k,
        },
    };
    var result = (await agent.Reader.ReadAsync(new[] { read }))[0];
    string Show(int id) => JsonSerializer.Serialize(agent.Reader.Tokenizer.Decode(new List<int> { id }));
    Console.WriteLine($"prompt {prompt.Length} tokens; tail: {JsonSerializer.Serialize(agent.Reader.Tokenizer.Decode(prompt.TakeLast(12).ToList()))}");
    var slots = schema.Slots.Select(s => s.Position).ToHashSet();
    for (int pos = 0; pos < canvas.Length; pos++)
    {
        var lp = result.Logprobs[pos];
        string top = string.Join("  ", lp.TokenIds.Select((id, i) => $"{Show(id)} {Math.Exp(lp.Logprobs[i]):F3}"));
        Console.WriteLine($"{pos,3}{(slots.Contains(pos) ? "*" : " ")} seed {Show(canvas[pos]),-16} -> {top}");
    }
    return 0;
}

static JevBenchSet LoadSet(CommandLine options)
{
    string root = options.Value("jevbench") ?? Environment.GetEnvironmentVariable("TS_JEVBENCH_DIR")
        ?? throw new ArgumentException("provide --jevbench <checkout> (or TS_JEVBENCH_DIR)");
    string[]? splits = options.Value("splits")?.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    JevBenchSet set = JevBenchSet.Load(root, splits);
    foreach (JevBenchSource source in set.Sources)
    {
        string check = source.MatchesManifest switch { true => "matches manifest", false => "DOES NOT match manifest", null => "not in manifest" };
        Console.Error.WriteLine($"{source.Split}: {source.Tasks} tasks, sha256 {source.Sha256[..12]}… ({check})");
    }
    return set;
}

static int Plan(CommandLine options)
{
    JevBenchSet set = LoadSet(options);
    using DiffusionAgent agent = OpenAgent(options);
    DecisionOptions decision = ReadOptions(options);
    int reads = 0, failures = 0;
    var widths = new SortedDictionary<int, int>();
    var prompts = new List<int>();
    foreach (JevBenchTask task in set.Tasks.Take(options.Int("limit") ?? int.MaxValue))
    {
        try
        {
            DecisionPlan plan = agent.Plan(task.ToRequest(decision));
            reads += plan.Reads;
            foreach (int w in plan.CanvasWidths) widths[w] = widths.GetValueOrDefault(w) + 1;
            prompts.AddRange(plan.PromptTokens);
        }
        catch (DecisionSchemaException ex)
        {
            failures++;
            Console.Error.WriteLine($"{task.Id}: {ex.Message}");
        }
    }
    prompts.Sort();
    Console.WriteLine($"{set.Tasks.Count} tasks, {reads} reads, {failures} that cannot compile");
    Console.WriteLine("canvas widths: " + string.Join(", ", widths.Select(w => $"{w.Key}×{w.Value}")));
    if (prompts.Count > 0)
        Console.WriteLine($"prompt tokens: median {prompts[prompts.Count / 2]}, max {prompts[^1]}, mean {prompts.Average():F0}");
    return failures == 0 ? 0 : 1;
}

static async Task<int> JevBench(CommandLine options)
{
    JevBenchSet set = LoadSet(options);
    using DiffusionAgent agent = OpenAgent(options);
    var runOptions = new JevBenchRunOptions
    {
        Decision = ReadOptions(options) with { Diagnostics = false },
        Pacing = options.Value("pacing") == "batched" ? JevBenchPacing.Batched : JevBenchPacing.Serial,
        Warmups = options.Int("warmups") ?? 2,
        Limit = options.Int("limit"),
        UsdPerMillionInputTokens = options.Double("price-per-m") ?? 0.035,
        PricingSource = options.Value("pricing-source") ?? "djev announced price, $0.035 per million input tokens, output free",
        DeviceUsdPerHour = options.Double("device-usd-per-hour"),
        EndpointKind = options.Value("endpoint") ?? "gpu",
        SystemName = options.Has("mock") ? "mock reader (harness check, no model)" : "TensorSharp DiffusionGemma (djev read)",
    };

    int done = 0, total = runOptions.Limit is { } l ? Math.Min(l, set.Tasks.Count) : set.Tasks.Count;
    var progress = new Progress<JevBenchOutcome>(o =>
    {
        done++;
        if (options.Has("verbose") || done % 25 == 0 || done == total)
            Console.Error.WriteLine(string.Format(CultureInfo.InvariantCulture, "[{0}/{1}] {2} {3} -> {4} ({5:F3} s){6}",
                done, total, o.Id, o.Expected, o.Predicted ?? "-", o.LatencyS, o.Error is null ? "" : " " + o.Error));
    });
    JevBenchReport report = await new JevBenchRunner(agent).RunAsync(set, runOptions, progress);

    string output = options.Value("out") ?? Path.Combine("results", $"jevbench-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, report.ToJson());

    Console.WriteLine($"{report.System} on {report.Decisions} public JevBench decisions ({report.Pacing})");
    foreach ((string tier, JevBenchTierSummary summary) in report.Tiers)
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "  {0,-9} {1,4}/{2,-4} {3,7:P1}  chance {4:P1}  above chance {5,5:F1}",
            tier, summary.Correct, summary.Scorable, summary.Accuracy ?? 0, summary.Chance, summary.ChanceCorrected ?? 0));
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "  accuracy {0:P1}, intelligence {1:F1} (published chance {2:F1}), calibration {3:F1} (ECE {4:F3}, TVD {5:F3})",
        report.Accuracy ?? 0, report.Intelligence ?? 0, report.IntelligencePublishedChance ?? 0,
        report.Calibration ?? 0, report.EceHard ?? 0, report.MeanTvdHard ?? 0));
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "  p50 {0:F3} s, p95 {1:F3} s, speed {2:F1} ({3}), {4:F1} decisions/s",
        report.P50S ?? 0, report.P95S ?? 0, report.Speed ?? 0, report.EndpointKind, report.DecisionsPerSecond));
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
        "  {0:F0} input tokens/decision, ${1:F4} per 1,000 decisions, cost {2:F1}; public-subset score {3:F1}",
        report.MeanInputTokens, report.UsdPer1000Decisions ?? 0, report.Cost ?? 0, report.PublicSubsetScore ?? 0));
    Console.WriteLine($"  {report.Failed} failed, {report.Valid} valid distributions; receipt: {output}");
    return 0;
}
