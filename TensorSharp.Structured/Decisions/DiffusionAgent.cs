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
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Structured.Decisions
{
    using JsonValue = System.Text.Json.Nodes.JsonValue;

    /// <summary>Engine-wide settings. The defaults are djev's served configuration.</summary>
    public sealed record DiffusionAgentOptions
    {
        /// <summary>Canvas geometry and token conventions.</summary>
        public DecisionCanvasOptions Canvas { get; init; } = new();

        /// <summary>The longest prompt plus canvas a read may use. Null takes the reader's context,
        /// which for DiffusionGemma is djev's <c>DJEV_MAX_MODEL_LEN</c> default, 32768.</summary>
        public int? MaxModelLength { get; init; }

        /// <summary>Canvases denoised together in one block. A read is one forward pass over its canvas,
        /// so a batch of them costs little more than one; too many runs the device out of memory.</summary>
        public int BatchSize { get; init; } = 16;

        /// <summary>The model name reported in results. Null uses the reader's.</summary>
        public string? ModelName { get; init; }
    }

    /// <summary>What a request will cost, without running it.</summary>
    public sealed record DecisionPlan(int Reads, IReadOnlyList<int> CanvasWidths, IReadOnlyList<int> PromptTokens,
        IReadOnlyList<string> SystemPrompts);

    /// <summary>
    /// Typed decisions on DiffusionGemma the way djev makes them: compile the questions into a system
    /// prompt and a one-token-per-question answer template, seed a compact canvas with that template and
    /// noise at the answer slots, run one denoising step, and read the temperature-1 probabilities of each
    /// question's allowed label tokens straight off the logits.
    ///
    /// <code>
    /// using var agent = DiffusionAgent.Load("diffusiongemma-26B-A4B-it-Q4_K_M.gguf", BackendType.GgmlCuda);
    /// var result = agent.SystemOne("I was charged twice and nobody answers", Presets.Triage());
    /// result["intent"].Choice;          // the most probable option
    /// result["frustration"].Score;      // the expected level
    /// result["refund_requested"].Noul;  // P(yes), relative to the two labels
    /// </code>
    ///
    /// Probabilities are relative to the allowed labels; <c>confidence</c> is entropy concentration. Neither
    /// is calibrated. One read is not the model's full iterative reasoning - it trades that for a small,
    /// fast decision.
    /// </summary>
    public sealed class DiffusionAgent : IDisposable
    {
        public const string ScoreEstimator = "truth-odds-v1";

        public const string ScoreLevelTask =
            "Determine whether the full description under yes is factually true of the state, " +
            "in the context of the evaluation instructions. Topic relevance alone is not enough.";

        public const string ScoreLevelFalse = "The candidate description is false of the state.";

        public const int MaxScoreModeReads = 128;

        private const string SeedPolicy = "djev-question-seed-v1";

        private readonly IDecisionReader _reader;
        private readonly DiffusionAgentOptions _options;
        private readonly DecisionSchemaCompiler _compiler;
        private readonly IDisposable? _owned;

        public DiffusionAgent(IDecisionReader reader, DiffusionAgentOptions? options = null)
            : this(reader, options, owned: null) { }

        private DiffusionAgent(IDecisionReader reader, DiffusionAgentOptions? options, IDisposable? owned)
        {
            _reader = reader ?? throw new ArgumentNullException(nameof(reader));
            _options = options ?? new DiffusionAgentOptions();
            if (_options.BatchSize < 1) throw new ArgumentOutOfRangeException(nameof(options), "BatchSize must be positive");
            _compiler = new DecisionSchemaCompiler(
                reader.Tokenizer, Math.Min(_options.Canvas.Canvas, reader.CanvasLength), _options.Canvas.Compact);
            _owned = owned;
        }

        /// <summary>Load a DiffusionGemma GGUF and answer with it. The agent owns the model.</summary>
        public static DiffusionAgent Load(string ggufPath, BackendType backend = BackendType.GgmlCpu,
            DiffusionAgentOptions? options = null)
        {
            ModelBase model = ModelBase.Create(ggufPath, backend);
            if (model is not DiffusionGemmaModel diffusion)
            {
                model.Dispose();
                throw new ArgumentException($"'{ggufPath}' is not a DiffusionGemma checkpoint", nameof(ggufPath));
            }
            return new DiffusionAgent(new DiffusionGemmaReader(diffusion), options, diffusion);
        }

        public IDecisionReader Reader => _reader;

        public string ModelName => _options.ModelName ?? _reader.ModelName;

        private int MaxModelLength => _options.MaxModelLength ?? _reader.MaxContextLength;

        private int NoiseVocabulary => _options.Canvas.NoiseVocabulary ?? _reader.VocabSize;

        // ---- Public API -------------------------------------------------------

        /// <summary>Answer every question about <paramref name="state"/> - a string, or anything that
        /// serializes to a JSON object or array.</summary>
        public DecisionResult SystemOne(object state, QuestionSet questions, DecisionOptions? options = null) =>
            SystemOneAsync(state, questions, options).GetAwaiter().GetResult();

        public Task<DecisionResult> SystemOneAsync(object state, QuestionSet questions,
            DecisionOptions? options = null, CancellationToken cancellationToken = default) =>
            PredictAsync(DecisionRequest.Create(state, questions, options), cancellationToken);

        /// <summary>Alias of <see cref="SystemOne"/>, for the Laya-style name.</summary>
        public DecisionResult Predict(object state, QuestionSet questions, DecisionOptions? options = null) =>
            SystemOne(state, questions, options);

        public async Task<DecisionResult> PredictAsync(DecisionRequest request, CancellationToken cancellationToken = default) =>
            (await PredictAsync(new[] { request }, cancellationToken).ConfigureAwait(false))[0];

        /// <summary>
        /// Answer several requests, batching their reads together up to
        /// <see cref="DiffusionAgentOptions.BatchSize"/> canvases per block. Each request is compiled and
        /// answered exactly as it would be alone; batching changes only how many canvases share a pass.
        /// </summary>
        public async Task<IReadOnlyList<DecisionResult>> PredictAsync(
            IReadOnlyList<DecisionRequest> requests, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(requests);
            var prepared = new List<Prepared>(requests.Count);
            foreach (DecisionRequest request in requests)
            {
                var watch = Stopwatch.StartNew();
                Prepared p = Prepare(request);
                p.CompileMs = watch.Elapsed.TotalMilliseconds;
                prepared.Add(p);
            }

            var jobs = prepared.SelectMany(p => p.Jobs).ToList();
            var results = new ReadEvidence[jobs.Count];
            var modelWatch = Stopwatch.StartNew();
            for (int offset = 0; offset < jobs.Count; offset += _options.BatchSize)
            {
                var chunk = jobs.Skip(offset).Take(_options.BatchSize).ToList();
                IReadOnlyList<DiffusionReadResult> reads = await _reader.ReadAsync(
                    chunk.Select(ToRead).ToList(), cancellationToken).ConfigureAwait(false);
                if (reads.Count != chunk.Count)
                    throw new DecisionBackendException("the reader returned a different number of reads than it was given");
                for (int i = 0; i < chunk.Count; i++) results[offset + i] = Evidence(chunk[i], reads[i]);
            }
            double modelMs = modelWatch.Elapsed.TotalMilliseconds;

            var answers = new List<DecisionResult>(prepared.Count);
            int cursor = 0;
            foreach (Prepared p in prepared)
            {
                var own = new ArraySegment<ReadEvidence>(results, cursor, p.Jobs.Count);
                cursor += p.Jobs.Count;
                answers.Add(p.Finish(own) with { CompileMs = p.CompileMs, ModelMs = modelMs });
            }
            return answers;
        }

        /// <summary>Compile a question set as a joint read would, to inspect its prompt, template and
        /// slots. Cached, like djev's schema cache - which holds schemas, never answers.</summary>
        public CompiledDecisionSchema Compile(QuestionSet questions) =>
            _compiler.Compile(questions.Select(q => q.Value).ToList());

        /// <summary>The reads a request would run, their widths and prompt lengths, without running them.</summary>
        public DecisionPlan Plan(DecisionRequest request)
        {
            Prepared p = Prepare(request);
            return new DecisionPlan(p.Jobs.Count,
                p.Jobs.Select(j => j.Schema.CanvasWidth).ToList(),
                p.Jobs.Select(j => j.PromptTokens.Length).ToList(),
                p.Jobs.Select(j => j.Schema.SystemPrompt).Distinct().ToList());
        }

        public void Dispose() => _owned?.Dispose();

        // ---- Preparation --------------------------------------------------------

        private sealed record Job(CompiledDecisionSchema Schema, int[] PromptTokens, BigInteger Seed);

        /// <summary>One read's evidence: per slot, the normalised label distribution, the raw label mass
        /// and the exact log-probabilities.</summary>
        private sealed record ReadEvidence(double[][] Probabilities, double[] Masses, double[][] Logprobs,
            int PromptTokens, int CompletionTokens);

        private sealed class Prepared
        {
            public required List<Job> Jobs { get; init; }
            public required Func<IReadOnlyList<ReadEvidence>, DecisionResult> Finish { get; init; }
            public double CompileMs { get; set; }
        }

        private Prepared Prepare(DecisionRequest request)
        {
            DecisionContracts.Validate(request);
            if (request.Options.ScoreMode == ScoreMode.IndependentLevels) return PrepareScoreLevels(request);
            if (request.Options.Isolation == DecisionIsolation.Independent) return PrepareIndependent(request);
            return PrepareJoint(request);
        }

        private static BigInteger BaseSeed(DecisionOptions options)
        {
            if (options.Seed is { } seed) return seed;
            Span<byte> bytes = stackalloc byte[8];
            RandomNumberGenerator.Fill(bytes);
            return new BigInteger(bytes, isUnsigned: true);
        }

        /// <summary>djev's <c>_independent_seed</c>: sha256 of the versioned seed record, as an integer.</summary>
        internal static BigInteger IndependentSeed(BigInteger baseSeed, string fingerprint, int sample)
        {
            var record = new JsonArray(SeedPolicy, baseSeed.ToString(CultureInfo.InvariantCulture), fingerprint, sample);
            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(PythonJson.Dumps(record)));
            return new BigInteger(hash, isUnsigned: true, isBigEndian: true);
        }

        private string StateText(DecisionRequest request) => DecisionSchemaCompiler.Describe(request.State);

        /// <summary>djev's <c>_check_context</c>: the prompt as the template renders it, and the canvas,
        /// must fit the model's context before any read is queued.</summary>
        private int[] CheckContext(CompiledDecisionSchema schema, string state)
        {
            int[] prompt = _reader.EncodeChat(schema.SystemPrompt, state);
            if (prompt.Length + schema.CanvasWidth > MaxModelLength)
                throw new DecisionSchemaException(
                    "the prompt and answer canvas exceed the model context limit; shorten the state or questions");
            return prompt;
        }

        private Prepared PrepareJoint(DecisionRequest request)
        {
            var questions = request.Questions.ToList();
            CompiledDecisionSchema schema = _compiler.Compile(questions.Select(q => q.Value).ToList());
            string state = StateText(request);
            int[] prompt = CheckContext(schema, state);
            BigInteger seed = BaseSeed(request.Options);
            int samples = request.Options.Samples;
            var jobs = Enumerable.Range(0, samples).Select(i => new Job(schema, prompt, seed + i * 7919)).ToList();

            return new Prepared
            {
                Jobs = jobs,
                Finish = reads =>
                {
                    var answers = new List<KeyValuePair<string, Answer>>();
                    var diagnostics = new JsonObject();
                    for (int qi = 0; qi < questions.Count; qi++)
                    {
                        double[] mean = Mean(reads, qi, schema.Slots[qi].TokenIds.Count);
                        answers.Add(new(questions[qi].Key, DecisionContracts.AnswerFrom(questions[qi].Value, mean)));
                        diagnostics[questions[qi].Key] = new JsonObject
                        {
                            ["label_mass"] = reads.Average(r => r.Masses[qi]),
                            ["label_entropy"] = DecisionContracts.Entropy(mean),
                        };
                    }
                    return Result(request, answers, reads, new[] { schema.CanvasWidth }, () => new JsonObject
                    {
                        ["engine"] = "diffusiongemma-tensorsharp",
                        ["calibration"] = "unvalidated",
                        ["probability_basis"] = "relative_to_allowed_labels",
                        ["isolation"] = "joint",
                        ["physical_reads"] = reads.Count,
                        ["samples"] = reads.Count,
                        ["steps"] = 1,
                        ["canvas_tokens"] = schema.CanvasWidth,
                        ["questions"] = diagnostics,
                    });
                },
            };
        }

        private Prepared PrepareIndependent(DecisionRequest request)
        {
            string state = StateText(request);
            // Identical questions share one group; choice order and score level order stay significant.
            var groups = new List<(string Key, CompiledDecisionSchema Schema, int[] Prompt, List<string> Ids)>();
            var byKey = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach ((string id, Question question) in request.Questions)
            {
                string key = DecisionSchemaCompiler.KeyOf(question);
                if (byKey.TryGetValue(key, out int existing)) { groups[existing].Ids.Add(id); continue; }
                CompiledDecisionSchema schema = _compiler.Compile(new[] { question });
                byKey[key] = groups.Count;
                groups.Add((key, schema, CheckContext(schema, state), new List<string> { id }));
            }

            BigInteger baseSeed = BaseSeed(request.Options);
            int count = request.Options.Samples;
            var jobs = new List<Job>();
            foreach (var group in groups)
            {
                string fingerprint = group.Schema.Fingerprint(group.Key);
                for (int sample = 0; sample < count; sample++)
                    jobs.Add(new Job(group.Schema, group.Prompt, IndependentSeed(baseSeed, fingerprint, sample)));
            }

            return new Prepared
            {
                Jobs = jobs,
                Finish = reads =>
                {
                    var answers = new Dictionary<string, Answer>(StringComparer.Ordinal);
                    var evidence = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
                    for (int g = 0; g < groups.Count; g++)
                    {
                        var samples = reads.Skip(g * count).Take(count).ToList();
                        double[] mean = Mean(samples, 0, groups[g].Schema.Slots[0].TokenIds.Count);
                        Answer answer = DecisionContracts.AnswerFrom(request.Questions[groups[g].Ids[0]], mean);
                        foreach (string id in groups[g].Ids)
                        {
                            answers[id] = answer;
                            evidence[id] = new JsonObject
                            {
                                ["label_mass"] = samples.Average(r => r.Masses[0]),
                                ["label_entropy"] = DecisionContracts.Entropy(mean),
                                ["canvas_tokens"] = groups[g].Schema.CanvasWidth,
                            };
                        }
                    }
                    var ordered = request.Questions.Ids.Select(id => new KeyValuePair<string, Answer>(id, answers[id])).ToList();
                    return Result(request, ordered, reads, groups.Select(g => g.Schema.CanvasWidth).ToList(), () => new JsonObject
                    {
                        ["engine"] = "diffusiongemma-tensorsharp",
                        ["calibration"] = "unvalidated",
                        ["probability_basis"] = "relative_to_allowed_labels",
                        ["isolation"] = "independent",
                        ["seed_policy"] = SeedPolicy,
                        ["samples"] = count,
                        ["steps"] = 1,
                        ["logical_questions"] = request.Questions.Count,
                        ["unique_questions"] = groups.Count,
                        ["physical_reads"] = reads.Count,
                        // djev fills this per group, so a duplicate sits next to the question it repeats.
                        ["questions"] = new JsonObject(groups.SelectMany(g => g.Ids).Select(
                            id => new KeyValuePair<string, JsonNode?>(id, evidence[id].DeepClone()))),
                    });
                },
            };
        }

        /// <summary>djev's experimental <c>independent_levels</c> Score mode: each level is its own Noul
        /// read, combined by the levels' log-odds (<c>truth-odds-v1</c>).</summary>
        private Prepared PrepareScoreLevels(DecisionRequest request)
        {
            string state = StateText(request);
            var children = new List<(string Key, Question Child)>();
            var childIndex = new Dictionary<string, int>(StringComparer.Ordinal);
            var mappings = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            var scoreKeys = new HashSet<string>(StringComparer.Ordinal);

            string Add(Question question)
            {
                string key = DecisionSchemaCompiler.KeyOf(question);
                if (!childIndex.ContainsKey(key))
                {
                    childIndex[key] = children.Count;
                    children.Add((key, question));
                }
                return key;
            }

            foreach ((string id, Question question) in request.Questions)
            {
                if (question.Type != QuestionType.Score) { mappings[id] = new List<string> { Add(question) }; continue; }
                var keys = new List<string>();
                foreach (JsonNode? level in question.Levels)
                {
                    if (level is null
                        || (level is JsonValue v && v.GetValueKind() == System.Text.Json.JsonValueKind.String
                            && string.IsNullOrWhiteSpace(v.GetValue<string>()))
                        || (level is JsonObject o && o.Count == 0)
                        || (level is JsonArray a && a.Count == 0))
                        throw new DecisionSchemaException("Independent Score levels require nonempty descriptions");
                    var child = Question.Noul(
                        new JsonObject { ["task"] = ScoreLevelTask, ["evaluation_instructions"] = question.Instructions?.DeepClone() },
                        trueCriterion: level.DeepClone(), falseCriterion: ScoreLevelFalse);
                    string key = Add(child);
                    keys.Add(key);
                    scoreKeys.Add(key);
                }
                mappings[id] = keys;
            }

            int count = request.Options.Samples;
            if (children.Count * count > MaxScoreModeReads)
                throw new DecisionSchemaException("Independent Score mode exceeds 128 physical reads; reduce questions, levels or samples");
            var schemas = children.Select(c =>
            {
                CompiledDecisionSchema schema = _compiler.Compile(new[] { c.Child });
                return (Schema: schema, Prompt: CheckContext(schema, state));
            }).ToList();

            BigInteger baseSeed = BaseSeed(request.Options);
            var jobs = new List<Job>();
            for (int c = 0; c < schemas.Count; c++)
            {
                (CompiledDecisionSchema schema, int[] prompt) = schemas[c];
                string fingerprint = schema.Fingerprint(children[c].Key);
                for (int sample = 0; sample < count; sample++)
                    jobs.Add(new Job(schema, prompt, IndependentSeed(baseSeed, fingerprint, sample)));
            }

            return new Prepared
            {
                Jobs = jobs,
                Finish = reads =>
                {
                    var means = new Dictionary<string, double[]>(StringComparer.Ordinal);
                    var evidence = new Dictionary<string, JsonObject>(StringComparer.Ordinal);
                    var odds = new Dictionary<string, double>(StringComparer.Ordinal);
                    for (int c = 0; c < children.Count; c++)
                    {
                        string key = children[c].Key;
                        var samples = reads.Skip(c * count).Take(count).ToList();
                        double[] mean = Mean(samples, 0, schemas[c].Schema.Slots[0].TokenIds.Count);
                        means[key] = mean;
                        evidence[key] = new JsonObject
                        {
                            ["label_mass"] = samples.Average(r => r.Masses[0]),
                            ["label_entropy"] = DecisionContracts.Entropy(mean),
                            ["canvas_tokens"] = schemas[c].Schema.CanvasWidth,
                        };
                        if (scoreKeys.Contains(key))
                        {
                            (double logNo, double logYes) = BinaryLogMeans(samples);
                            double logOdds = logYes - logNo;
                            odds[key] = logOdds;
                            evidence[key]["support"] = Math.Exp(logYes);
                            evidence[key]["log_mean_yes"] = JsonLog(logYes);
                            evidence[key]["log_mean_no"] = JsonLog(logNo);
                            evidence[key]["log_odds"] = JsonLog(logOdds);
                        }
                    }

                    var answers = new List<KeyValuePair<string, Answer>>();
                    var diagnostics = new JsonObject();
                    foreach ((string id, Question question) in request.Questions)
                    {
                        List<string> keys = mappings[id];
                        if (question.Type == QuestionType.Score)
                        {
                            double[] probabilities = OddsDistribution(keys.Select(k => odds[k]).ToList());
                            answers.Add(new(id, DecisionContracts.AnswerFrom(question, probabilities)));
                            diagnostics[id] = new JsonObject
                            {
                                ["estimator"] = ScoreEstimator,
                                ["label_entropy"] = DecisionContracts.Entropy(probabilities),
                                ["levels"] = new JsonObject(keys.Select((k, i) => new KeyValuePair<string, JsonNode?>(
                                    i.ToString(CultureInfo.InvariantCulture), evidence[k].DeepClone()))),
                            };
                        }
                        else
                        {
                            answers.Add(new(id, DecisionContracts.AnswerFrom(question, means[keys[0]])));
                            diagnostics[id] = evidence[keys[0]].DeepClone();
                        }
                    }
                    return Result(request, answers, reads, schemas.Select(s => s.Schema.CanvasWidth).ToList(), () => new JsonObject
                    {
                        ["engine"] = "diffusiongemma-tensorsharp",
                        ["calibration"] = "unvalidated",
                        ["probability_basis"] = "conditional_bernoulli_odds_for_scores",
                        ["isolation"] = "independent",
                        ["score_mode"] = "independent_levels",
                        ["estimator"] = ScoreEstimator,
                        ["seed_policy"] = SeedPolicy,
                        ["samples"] = count,
                        ["steps"] = 1,
                        ["logical_questions"] = request.Questions.Count,
                        ["logical_score_levels"] = request.Questions.Where(q => q.Value.Type == QuestionType.Score)
                            .Sum(q => mappings[q.Key].Count),
                        ["unique_children"] = children.Count,
                        ["physical_reads"] = reads.Count,
                        ["questions"] = diagnostics,
                    });
                },
            };
        }

        // ---- Reads ------------------------------------------------------------------

        private DecisionRead ToRead(Job job)
        {
            CompiledDecisionSchema schema = job.Schema;
            var requested = new int[schema.CanvasWidth][];
            foreach (DecisionSlot slot in schema.Slots) requested[slot.Position] = slot.TokenIds.ToArray();
            return new DecisionRead
            {
                PromptTokens = job.PromptTokens,
                Options = new DiffusionReadOptions
                {
                    ReadOnly = true,
                    MaxSteps = 1,
                    CanvasWidth = schema.CanvasWidth,
                    SeedCanvas = DecisionSchemaCompiler.SeedCanvas(schema, job.Seed, NoiseVocabulary,
                        _options.Canvas.TerminatorTokenId, _options.Canvas.PadTokenId),
                    LogprobTokenIds = requested,
                },
            };
        }

        /// <summary>
        /// Validate and normalise one read's evidence, as djev's <c>_read</c> does: every requested label
        /// must be reported, log-probabilities must be finite-or-impossible and not above zero, and the
        /// distribution is formed over the allowed labels only.
        /// </summary>
        private static ReadEvidence Evidence(Job job, DiffusionReadResult read)
        {
            var slots = job.Schema.Slots;
            var probabilities = new double[slots.Count][];
            var masses = new double[slots.Count];
            var logprobs = new double[slots.Count][];
            for (int s = 0; s < slots.Count; s++)
            {
                DecisionSlot slot = slots[s];
                if (read.Logprobs.Count <= slot.Position)
                    throw new DecisionBackendException("the inference backend omitted an exact label log probability");
                DiffusionPositionLogprobs row = read.Logprobs[slot.Position];
                var scores = new Dictionary<int, double>();
                for (int i = 0; i < row.TokenIds.Length; i++)
                {
                    double value = row.Logprobs[i];
                    if (double.IsNaN(value) || double.IsPositiveInfinity(value) || value > 1e-5)
                        throw new DecisionBackendException("the inference backend returned invalid log probabilities");
                    // vLLM serializes -inf as -9999; either is an impossible label, never finite evidence.
                    if (value <= -9999.0) value = double.NegativeInfinity;
                    if (scores.TryGetValue(row.TokenIds[i], out double seen) && seen != value)
                        throw new DecisionBackendException("the inference backend returned conflicting log probabilities");
                    scores[row.TokenIds[i]] = value;
                }
                if (slot.TokenIds.Any(id => !scores.ContainsKey(id)))
                    throw new DecisionBackendException("the inference backend omitted an exact label log probability");
                double[] exact = slot.TokenIds.Select(id => scores[id]).ToArray();
                probabilities[s] = DecisionContracts.NormalizeLogprobs(exact);
                masses[s] = DecisionContracts.Fsum(exact.Select(Math.Exp).ToArray());
                logprobs[s] = exact;
            }
            return new ReadEvidence(probabilities, masses, logprobs, job.PromptTokens.Length, job.Schema.Template.Count + 1);
        }

        private static double[] Mean(IReadOnlyList<ReadEvidence> reads, int slot, int labels)
        {
            var mean = new double[labels];
            for (int li = 0; li < labels; li++)
                mean[li] = DecisionContracts.Fsum(reads.Select(r => r.Probabilities[slot][li]).ToArray()) / reads.Count;
            return mean;
        }

        private DecisionResult Result(DecisionRequest request, List<KeyValuePair<string, Answer>> answers,
            IReadOnlyList<ReadEvidence> reads, IReadOnlyList<int> widths, Func<JsonObject> diagnostics) => new()
        {
            Model = ModelName,
            Answers = answers,
            Usage = new DecisionUsage(reads.Sum(r => r.PromptTokens), reads.Sum(r => r.CompletionTokens)),
            Diagnostics = request.Options.Diagnostics ? diagnostics() : null,
            CanvasWidths = widths,
        };

        // ---- truth-odds-v1 -------------------------------------------------------------

        private static double LogSumExp(IReadOnlyList<double> values)
        {
            double peak = values.Max();
            if (double.IsNegativeInfinity(peak)) return peak;
            return peak + Math.Log(DecisionContracts.Fsum(values.Select(v => Math.Exp(v - peak)).ToArray()));
        }

        /// <summary>Average both conditional binary probabilities in log space, without rounding a small
        /// complement to 0 or 1.</summary>
        private static (double LogNo, double LogYes) BinaryLogMeans(IReadOnlyList<ReadEvidence> reads)
        {
            var noLogs = new List<double>();
            var yesLogs = new List<double>();
            foreach (ReadEvidence read in reads)
            {
                double no = read.Logprobs[0][0], yes = read.Logprobs[0][1];
                double correction = Log1P(Math.Exp(-Math.Abs(yes - no)));
                if (no >= yes)
                {
                    noLogs.Add(-correction);
                    yesLogs.Add(yes - no - correction);
                }
                else
                {
                    noLogs.Add(no - yes - correction);
                    yesLogs.Add(-correction);
                }
            }
            double divisor = Math.Log(reads.Count);
            return (LogSumExp(noLogs) - divisor, LogSumExp(yesLogs) - divisor);
        }

        /// <summary><c>log(1 + x)</c> without losing a small <paramref name="x"/> to the addition.</summary>
        private static double Log1P(double x)
        {
            double u = 1.0 + x;
            return u == 1.0 ? x : Math.Log(u) * x / (u - 1.0);
        }

        private static double[] OddsDistribution(IReadOnlyList<double> logOdds)
        {
            var certain = Enumerable.Range(0, logOdds.Count).Where(i => double.IsPositiveInfinity(logOdds[i])).ToList();
            if (certain.Count > 1 || logOdds.All(double.IsNegativeInfinity))
                throw new DecisionBackendException("Score exactly-one conditioning has zero evidence");
            if (certain.Count == 1)
                return Enumerable.Range(0, logOdds.Count).Select(i => i == certain[0] ? 1.0 : 0.0).ToArray();
            return DecisionContracts.NormalizeLogprobs(logOdds);
        }

        private static JsonNode JsonLog(double value) =>
            double.IsPositiveInfinity(value) ? JsonValue.Create("+inf")
            : double.IsNegativeInfinity(value) ? JsonValue.Create("-inf")
            : JsonValue.Create(value);
    }
}
