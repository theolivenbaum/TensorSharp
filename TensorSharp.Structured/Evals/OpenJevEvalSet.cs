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
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Evals
{
    /// <summary>How many questions a workflow contributes, and how many of them can be scored.</summary>
    public sealed record OpenJevWorkflowCounts(
        string Workflow, int Questions, int Scorable, int MissingReference, int ReferenceTie);

    /// <summary>
    /// Reads the public evaluation materials that
    /// <see href="https://github.com/theolivenbaum/open-jev">open-jev</see> carries, so the same questions
    /// can be replayed through <see cref="StructuredBenchmark"/> and the receipts put side by side.
    ///
    /// <b>The data is not redistributed here.</b> It is third-party evaluation material with its own
    /// rights, published under the source URLs and hashes in open-jev's <c>manifest.json</c>; point this
    /// at a checkout rather than copying the files into this repository. <see cref="SourceSha256"/>
    /// records what was actually read, so a receipt names the inputs it was produced from.
    ///
    /// Scoring follows open-jev exactly, which is what makes the two comparable: a question's reference is
    /// the consensus of its adjudication sets, averaged, and it is scored only when that consensus has a
    /// single winner. Questions with no reference, or with a tie, are carried as
    /// <see cref="StructuredBenchmarkCase.Unscored"/> rather than dropped. The saved answers of the system
    /// the set was published for are carried as <see cref="StructuredBenchmarkCase.Baseline"/> and scored
    /// on the same questions.
    /// </summary>
    public static class OpenJevEvalSet
    {
        /// <summary>The loaded set: one case per question node, plus what it was read from.</summary>
        public sealed record Result(
            IReadOnlyList<StructuredBenchmarkCase> Cases,
            IReadOnlyDictionary<string, string> SourceSha256,
            IReadOnlyList<OpenJevWorkflowCounts> Counts)
        {
            public int Questions => Counts.Sum(c => c.Questions);
            public int Scorable => Counts.Sum(c => c.Scorable);
        }

        private static readonly JsonSerializerOptions DocumentJson = new()
        {
            // The documents are shown to the model, so they keep their own characters rather than being
            // escaped into \uXXXX - which is also what open-jev's ensure_ascii=False does.
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            WriteIndented = false,
        };

        /// <summary>
        /// Load every workflow in <paramref name="directory"/> (open-jev's <c>typesafe-public-evals</c>
        /// folder, or a checkout root containing it).
        /// </summary>
        /// <exception cref="DirectoryNotFoundException">No such folder.</exception>
        /// <exception cref="InvalidDataException">A workflow file is not in the expected shape.</exception>
        public static Result Load(string directory)
        {
            ArgumentException.ThrowIfNullOrEmpty(directory);
            string root = Directory.Exists(Path.Combine(directory, "typesafe-public-evals"))
                ? Path.Combine(directory, "typesafe-public-evals")
                : directory;
            if (!Directory.Exists(root))
                throw new DirectoryNotFoundException($"No public-eval folder at '{directory}'.");

            var cases = new List<StructuredBenchmarkCase>();
            var sources = new Dictionary<string, string>(StringComparer.Ordinal);
            var counts = new List<OpenJevWorkflowCounts>();

            foreach (string path in Directory.GetFiles(root, "*.json").OrderBy(p => p, StringComparer.Ordinal))
            {
                string name = Path.GetFileName(path);
                if (name == "manifest.json") continue;
                sources[name] = Sha256(path);
                counts.Add(LoadWorkflow(path, cases));
            }
            if (cases.Count == 0)
                throw new InvalidDataException($"No workflow files found under '{root}'.");
            return new Result(cases, sources, counts);
        }

        private static OpenJevWorkflowCounts LoadWorkflow(string path, List<StructuredBenchmarkCase> into)
        {
            string workflow = Path.GetFileNameWithoutExtension(path);
            JsonObject data = JsonNode.Parse(File.ReadAllText(path)) as JsonObject
                ?? throw new InvalidDataException($"'{path}' is not a JSON object.");
            JsonArray questionBank = Array(data, "questions", path);
            JsonArray documents = Array(data, "documents", path);
            JsonObject allCases = Object(data, "cases", path);

            int questions = 0, scorable = 0, missing = 0, tie = 0;
            foreach ((string caseId, JsonNode? caseNode) in allCases)
            {
                JsonObject theCase = caseNode as JsonObject
                    ?? throw new InvalidDataException($"Case '{caseId}' in '{path}' is not an object.");
                JsonObject published = Object(Object(theCase, "models", path), "typesafe", path);
                JsonObject? referenceAnswers = theCase["reference_answers"] as JsonObject;

                foreach (JsonNode? nodeNode in Array(published, "nodes", path))
                {
                    if (nodeNode is not JsonObject node) continue;
                    // A node the published run skipped has no saved answers to compare against.
                    if (node["ran"]?.GetValue<bool>() != true) continue;

                    string nodeName = node["node"]!.GetValue<string>();
                    JsonObject nodeQuestions = Object(node, "questions", path);
                    JsonObject answers = Object(node, "answers", path);
                    JsonObject? nodeReferences = referenceAnswers?[nodeName] as JsonObject;

                    var schema = new Dictionary<string, StructuredQuestion>();
                    var expected = new Dictionary<string, object?>();
                    var unscored = new Dictionary<string, string>();
                    var baseline = new Dictionary<string, object?>();

                    foreach ((string key, JsonNode? indexNode) in nodeQuestions)
                    {
                        JsonObject question = questionBank[indexNode!.GetValue<int>()] as JsonObject
                            ?? throw new InvalidDataException($"Bad question index for '{key}' in '{path}'.");
                        StructuredQuestion structured = ToQuestion(question, path);
                        schema[key] = structured;
                        questions++;

                        (object? reference, string? why) =
                            Consensus(nodeReferences?[key] as JsonObject, structured);
                        if (reference is not null) { expected[key] = reference; scorable++; }
                        else
                        {
                            string reason = why ?? "missing_reference";
                            unscored[key] = reason;
                            if (reason == "reference_tie") tie++;
                            else if (reason == "missing_reference") missing++;
                        }

                        if (answers[key] is JsonObject saved
                            && SavedAnswer(saved, structured) is { } savedValue)
                            baseline[key] = savedValue;
                    }
                    if (schema.Count == 0) continue;

                    into.Add(new StructuredBenchmarkCase
                    {
                        Workflow = workflow,
                        Request = new StructuredRequest
                        {
                            Id = string.Join('/', workflow, caseId, nodeName),
                            Document = JsonSerializer.Serialize(
                                documents[node["doc"]!.GetValue<int>()], DocumentJson),
                            Questions = schema,
                        },
                        Expected = expected,
                        Unscored = unscored,
                        Baseline = baseline,
                    });
                }
            }
            return new OpenJevWorkflowCounts(workflow, questions, scorable, missing, tie);
        }

        /// <summary>A question's allowed values and the wording of each, by type: a yes/no, a pick-one over
        /// named criteria, or an ordered scale answered by its zero-based index.</summary>
        private static StructuredQuestion ToQuestion(JsonObject question, string path)
        {
            string type = question["type"]?.GetValue<string>()
                ?? throw new InvalidDataException($"A question in '{path}' has no type.");
            string instructions = question["instructions"]?.GetValue<string>() ?? "";
            JsonNode? criteria = question["criteria"];

            switch (type)
            {
                case "noul":
                    return new StructuredQuestion
                    {
                        Instructions = instructions,
                        Options = new object?[] { false, true },
                        Criteria = criteria is JsonObject flags
                            ? new[] { Text(flags["false"]), Text(flags["true"]) }
                            : null,
                        Kind = type,
                    };
                case "score":
                    var levels = (criteria as JsonArray)?.Select(Text).ToArray()
                        ?? throw new InvalidDataException($"A score question in '{path}' has no levels.");
                    return new StructuredQuestion
                    {
                        Instructions = instructions,
                        Options = Enumerable.Range(0, levels.Length).Cast<object?>().ToArray(),
                        Criteria = levels,
                        Kind = type,
                    };
                case "choice":
                    var options = criteria as JsonObject
                        ?? throw new InvalidDataException($"A choice question in '{path}' has no options.");
                    return new StructuredQuestion
                    {
                        Instructions = instructions,
                        Options = options.Select(x => (object?)x.Key).ToArray(),
                        Criteria = options.Select(x => Text(x.Value)).ToArray(),
                        Kind = type,
                    };
                default:
                    throw new InvalidDataException($"Unsupported question type '{type}' in '{path}'.");
            }
        }

        /// <summary>
        /// The reference answer: each adjudication set's distribution averaged over the sets, then the
        /// single highest value - or null when the sets tie, which is not a reference to score against.
        /// A set that recorded only a value counts as all of its mass.
        /// </summary>
        private static (object? Reference, string? Why) Consensus(
            JsonObject? reference, StructuredQuestion question)
        {
            if (reference?["sets"] is not JsonArray sets || sets.Count == 0)
                return (null, "missing_reference");

            var mass = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (JsonNode? setNode in sets)
            {
                if (setNode is not JsonObject set) continue;
                if (set["probabilities"] is JsonObject probabilities)
                {
                    foreach ((string label, JsonNode? p) in probabilities)
                        mass[label] = mass.GetValueOrDefault(label) + p!.GetValue<double>() / sets.Count;
                }
                else if (set["value"] is { } value)
                {
                    string label = Canonical(value);
                    mass[label] = mass.GetValueOrDefault(label) + 1.0 / sets.Count;
                }
            }
            if (Winner(mass) is not { } winner) return (null, "reference_tie");
            // A label the question has no value for is a data problem, and saying so beats inflating the
            // tie count with it.
            return MatchOption(winner, question) is { } matched
                ? (matched, null)
                : (null, "unmatched_reference");
        }

        /// <summary>The saved answer of the system this set was published for, in the question's own type.
        /// A yes/no is saved as a probability; a choice as the chosen label.</summary>
        private static object? SavedAnswer(JsonObject answer, StructuredQuestion question)
        {
            if (question.Kind == "noul")
                return answer["noul"] is { } p ? p.GetValue<double>() >= 0.5 : null;

            // A recorded distribution decides, and a tie in it is no answer - the saved choice is a
            // fallback for an answer that carries no distribution at all, not a tie-breaker for one.
            if (answer["probabilities"] is JsonObject probabilities && probabilities.Count > 0)
            {
                return Winner(probabilities.ToDictionary(x => x.Key, x => x.Value!.GetValue<double>()))
                    is { } best ? MatchOption(best, question) : null;
            }
            return answer["choice"] is { } choice ? MatchOption(Canonical(choice), question) : null;
        }

        /// <summary>The single largest entry, or null when nothing leads outright.</summary>
        private static string? Winner(IReadOnlyDictionary<string, double> mass)
        {
            if (mass.Count == 0) return null;
            double max = mass.Values.Max();
            string[] leaders = mass.Where(x => Math.Abs(x.Value - max) < 1e-9).Select(x => x.Key).ToArray();
            return leaders.Length == 1 ? leaders[0] : null;
        }

        /// <summary>
        /// Turn a reference's label back into the question's own allowed value. References are written as
        /// strings ("3", "true", "refund_request") whatever the question answers in, so this matches by
        /// the same canonical spelling rather than trusting the type.
        /// </summary>
        private static object? MatchOption(string label, StructuredQuestion question)
        {
            foreach (object? option in question.Options)
            {
                if (string.Equals(CanonicalOf(option), label, StringComparison.OrdinalIgnoreCase))
                    return option;
            }
            return null;
        }

        private static string Canonical(JsonNode value) => value.GetValueKind() switch
        {
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => value.ToString(),
        };

        private static string CanonicalOf(object? option) => option switch
        {
            bool b => b ? "true" : "false",
            null => "null",
            _ => option.ToString() ?? "",
        };

        private static string Text(JsonNode? node) => node?.GetValue<string>() ?? "";

        private static JsonArray Array(JsonObject parent, string name, string path) =>
            parent[name] as JsonArray
            ?? throw new InvalidDataException($"'{path}' has no '{name}' array.");

        private static JsonObject Object(JsonObject parent, string name, string path) =>
            parent[name] as JsonObject
            ?? throw new InvalidDataException($"'{path}' has no '{name}' object.");

        private static string Sha256(string path)
        {
            using FileStream stream = File.OpenRead(path);
            return Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(stream));
        }
    }
}
