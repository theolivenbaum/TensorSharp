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
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Structured.Decisions;

namespace TensorSharp.Structured.Evals
{
    using JsonValue = System.Text.Json.Nodes.JsonValue;

    /// <summary>
    /// One JevBench decision: a state, one typed question, the exact label set and the reference answer.
    /// </summary>
    public sealed record JevBenchTask
    {
        public required string Id { get; init; }

        /// <summary>The scoring tier: <c>easy</c>, <c>standard</c>, <c>judge</c> or <c>hard</c>.</summary>
        public required string Tier { get; init; }

        /// <summary>The split file the task came from (<c>original</c>, <c>easy</c>, <c>hard</c>, ...).</summary>
        public required string Split { get; init; }

        public required string Family { get; init; }

        public required JsonNode State { get; init; }

        /// <summary>The question exactly as jevbench stores it - what its djev adapter sends.</summary>
        public required JsonObject QuestionJson { get; init; }

        public required Question Question { get; init; }

        /// <summary>The exact labels a distribution must cover: <c>no, yes</c>; the option names; or
        /// <c>"0".."n-1"</c>.</summary>
        public required IReadOnlyList<string> Labels { get; init; }

        /// <summary>The reference label, or null where jevbench has none (excluded from accuracy).</summary>
        public string? Expected { get; init; }

        /// <summary>Hard tier probability items: the exact gold distribution, for the TVD half of
        /// Calibration.</summary>
        public IReadOnlyDictionary<string, double>? GoldProbs { get; init; }

        /// <summary>Paraphrase group, when the task has a paired paraphrase.</summary>
        public string? Group { get; init; }

        /// <summary>What uniform guessing scores on this task: <c>1 / labels</c>.</summary>
        public double Chance => 1.0 / Labels.Count;

        /// <summary>The request jevbench's djev adapter makes for this task: the state, and the question
        /// under the fixed key <c>decision</c>.</summary>
        public DecisionRequest ToRequest(DecisionOptions? options = null) => new()
        {
            Id = Id,
            State = State.DeepClone(),
            Questions = new QuestionSet().Add(JevBenchSet.QuestionKey, Question),
            Options = options ?? DecisionOptions.Default,
        };
    }

    /// <summary>A split file as read: its name, its hash, and whether the hash matches the manifest.</summary>
    public sealed record JevBenchSource(
        [property: System.Text.Json.Serialization.JsonPropertyName("split")] string Split,
        [property: System.Text.Json.Serialization.JsonPropertyName("path")] string Path,
        [property: System.Text.Json.Serialization.JsonPropertyName("tasks")] int Tasks,
        [property: System.Text.Json.Serialization.JsonPropertyName("sha256")] string Sha256,
        [property: System.Text.Json.Serialization.JsonPropertyName("matches_manifest")] bool? MatchesManifest);

    /// <summary>
    /// JevBench's public decisions (<c>datasets/public/*.jsonl</c> in a jevbench checkout), read into
    /// tasks the way jevbench itself reads them.
    ///
    /// The data is not vendored: point the loader at a checkout. Every file's SHA-256 is compared with
    /// jevbench's <c>datasets/manifest.json</c>, so a receipt names exactly what it ran on. Held-out items
    /// are not public; a run on this set measures the public items only, and is not a JevBench Score.
    /// </summary>
    public sealed class JevBenchSet
    {
        /// <summary>The question key jevbench's adapters use.</summary>
        public const string QuestionKey = "decision";

        public IReadOnlyList<JevBenchTask> Tasks { get; }

        public IReadOnlyList<JevBenchSource> Sources { get; }

        public string? Protocol { get; }

        private JevBenchSet(IReadOnlyList<JevBenchTask> tasks, IReadOnlyList<JevBenchSource> sources, string? protocol)
        {
            Tasks = tasks;
            Sources = sources;
            Protocol = protocol;
        }

        /// <summary>
        /// Load a jevbench checkout (or its <c>datasets</c> directory, or a directory of split files).
        /// </summary>
        /// <param name="root">The checkout.</param>
        /// <param name="splits">Split names to read (<c>easy</c>, <c>original</c>, <c>hard</c>, ...); null
        /// reads every <c>*.jsonl</c> in the public directory.</param>
        public static JevBenchSet Load(string root, IEnumerable<string>? splits = null)
        {
            string datasets = Directory.Exists(Path.Combine(root, "datasets")) ? Path.Combine(root, "datasets") : root;
            string publicDir = Directory.Exists(Path.Combine(datasets, "public")) ? Path.Combine(datasets, "public") : datasets;
            if (!Directory.Exists(publicDir))
                throw new DirectoryNotFoundException($"no jevbench dataset directory under '{root}'");

            var manifestHashes = new Dictionary<string, string>(StringComparer.Ordinal);
            var tierOf = new Dictionary<string, string>(StringComparer.Ordinal) { ["hard"] = "hard" };
            string? protocol = null;
            string manifestPath = Path.Combine(datasets, "manifest.json");
            if (File.Exists(manifestPath))
            {
                JsonNode manifest = JsonNode.Parse(File.ReadAllText(manifestPath))!;
                protocol = manifest["protocol"]?.GetValue<string>();
                foreach (JsonNode? split in manifest["splits"]?.AsArray() ?? new JsonArray())
                {
                    if (split?["name"]?.GetValue<string>() is { } name && split["sha256"]?.GetValue<string>() is { } sha)
                        manifestHashes[name] = sha;
                }
                if (manifest["tiers"] is JsonObject tiers)
                {
                    foreach ((string tier, JsonNode? names) in tiers)
                        foreach (JsonNode? name in names?.AsArray() ?? new JsonArray())
                            if (name is not null) tierOf[name.GetValue<string>()] = tier;
                }
            }

            IEnumerable<string> files = splits is null
                ? Directory.GetFiles(publicDir, "*.jsonl").OrderBy(f => f, StringComparer.Ordinal)
                : splits.Select(s => Path.Combine(publicDir, s + ".jsonl"));

            var tasks = new List<JevBenchTask>();
            var sources = new List<JevBenchSource>();
            foreach (string file in files)
            {
                if (!File.Exists(file)) throw new FileNotFoundException($"jevbench split not found: {file}", file);
                string split = Path.GetFileNameWithoutExtension(file);
                byte[] raw = File.ReadAllBytes(file);
                string sha = Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant();
                string tier = tierOf.GetValueOrDefault(split, split);
                int before = tasks.Count;
                using var reader = new StreamReader(new MemoryStream(raw));
                int lineNumber = 0;
                while (reader.ReadLine() is { } line)
                {
                    lineNumber++;
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    try
                    {
                        tasks.Add(Parse(JsonNode.Parse(line)!.AsObject(), split, tier));
                    }
                    catch (Exception ex) when (ex is JsonException or InvalidOperationException or FormatException or ArgumentException)
                    {
                        throw new FormatException($"{file}:{lineNumber}: {ex.Message}", ex);
                    }
                }
                sources.Add(new JevBenchSource(split, file, tasks.Count - before, sha,
                    manifestHashes.TryGetValue(split, out string? expected) ? expected == sha : null));
            }
            return new JevBenchSet(tasks, sources, protocol);
        }

        private static JevBenchTask Parse(JsonObject record, string split, string tier)
        {
            var question = (JsonObject)record["question"]!.DeepClone();
            // jevbench's build_question: type and instructions, criteria only when present.
            var sent = new JsonObject
            {
                ["type"] = question["type"]!.GetValue<string>(),
                ["instructions"] = question["instructions"]?.DeepClone(),
            };
            if (question["criteria"] is { } criteria) sent["criteria"] = criteria.DeepClone();

            JsonNode? expected = record["expected"];
            JsonObject? gold = record["provenance"]?["gold_probs"] as JsonObject;
            return new JevBenchTask
            {
                Id = record["id"]!.GetValue<string>(),
                Tier = tier,
                Split = split,
                Family = record["family"]?.GetValue<string>() ?? "unknown",
                State = record["state"]!.DeepClone(),
                QuestionJson = sent,
                Question = Question.FromJson(sent),
                Labels = record["labels"]!.AsArray().Select(l => l!.ToString()).ToList(),
                Expected = expected switch
                {
                    null => null,
                    JsonValue v when v.GetValueKind() == JsonValueKind.String => v.GetValue<string>(),
                    _ => expected.ToJsonString(),
                },
                GoldProbs = gold?.ToDictionary(p => p.Key, p => p.Value!.GetValue<double>()),
                Group = record["group"]?.GetValue<string>(),
            };
        }

        /// <summary>The tiers present, in jevbench's weighting order.</summary>
        public IReadOnlyList<string> Tiers =>
            JevBenchScoring.TierWeights.Keys.Where(t => Tasks.Any(x => x.Tier == t))
                .Concat(Tasks.Select(t => t.Tier).Distinct().Where(t => !JevBenchScoring.TierWeights.ContainsKey(t)))
                .ToList();

        public override string ToString() =>
            string.Join(", ", Sources.Select(s => string.Create(CultureInfo.InvariantCulture, $"{s.Split}: {s.Tasks}")));
    }
}
