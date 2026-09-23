// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>How questions share reads.</summary>
    public enum DecisionIsolation
    {
        /// <summary>Every question on one canvas, answered by one read (per sample). Questions attend to
        /// each other.</summary>
        Joint,

        /// <summary>One read per distinct question, each with a seed derived from its own content, so an
        /// unrelated question cannot change the focal one's prompt. More physical work.</summary>
        Independent,
    }

    /// <summary>How Score questions are read.</summary>
    public enum ScoreMode
    {
        /// <summary>One slot whose labels are the level indices.</summary>
        Categorical,

        /// <summary>Experimental: each level becomes its own yes/no read ("is this level's description
        /// true of the state?") and the levels are combined by their odds (djev's
        /// <c>truth-odds-v1</c>). Needs <see cref="DecisionIsolation.Independent"/>.</summary>
        IndependentLevels,
    }

    /// <summary>
    /// Per-request inference options - djev's <c>options</c> object. Defaults match djev's: one sample,
    /// one step, seed 0, joint isolation, categorical scores.
    /// </summary>
    public sealed record DecisionOptions
    {
        /// <summary>1-4 independent one-step reads, whose label distributions are averaged. They are not
        /// extra denoising steps on the same canvas.</summary>
        public int Samples { get; init; } = 1;

        /// <summary>The initial canvas seed; null draws fresh noise. djev seeds are integers of any size,
        /// hence <see cref="BigInteger"/>. A fixed seed fixes the canvas, not the arithmetic.</summary>
        public BigInteger? Seed { get; init; } = BigInteger.Zero;

        public DecisionIsolation Isolation { get; init; } = DecisionIsolation.Joint;

        public ScoreMode ScoreMode { get; init; } = ScoreMode.Categorical;

        /// <summary>Add djev's diagnostics block to the result.</summary>
        public bool Diagnostics { get; init; }

        public static DecisionOptions Default { get; } = new();

        internal void Validate()
        {
            if (Samples is < 1 or > 4) throw new DecisionSchemaException("samples must be between 1 and 4");
            if (ScoreMode == ScoreMode.IndependentLevels && Isolation != DecisionIsolation.Independent)
                throw new DecisionSchemaException("independent_levels requires isolation=independent");
        }

        /// <summary>djev's <c>options</c> object, every field spelled out.</summary>
        public JsonObject ToJson() => new()
        {
            ["samples"] = Samples,
            ["steps"] = 1,
            ["diagnostics"] = Diagnostics,
            ["seed"] = Seed is { } seed ? JsonNode.Parse(seed.ToString(CultureInfo.InvariantCulture)) : null,
            ["isolation"] = Isolation == DecisionIsolation.Joint ? "joint" : "independent",
            ["score_mode"] = ScoreMode == ScoreMode.Categorical ? "categorical" : "independent_levels",
        };

        /// <summary>djev's <c>options</c> object. <c>steps</c> must be 1, as there.</summary>
        public static DecisionOptions FromJson(JsonNode? node)
        {
            if (node is null) return Default;
            if (node is not JsonObject obj) throw new DecisionSchemaException("options must be a JSON object");
            var options = new DecisionOptions();
            foreach ((string key, JsonNode? value) in obj)
            {
                options = key switch
                {
                    "samples" => options with { Samples = value?.GetValue<int>() ?? 1 },
                    "steps" => (value?.ToJsonString() == "1") ? options
                        : throw new DecisionSchemaException("steps must be the integer 1"),
                    "diagnostics" => options with { Diagnostics = value?.GetValue<bool>() ?? false },
                    "seed" => options with
                    {
                        Seed = value is null ? null : BigInteger.Parse(value.ToJsonString(), CultureInfo.InvariantCulture),
                    },
                    "isolation" => options with
                    {
                        Isolation = value?.GetValue<string>() switch
                        {
                            "joint" => DecisionIsolation.Joint,
                            "independent" => DecisionIsolation.Independent,
                            _ => throw new DecisionSchemaException("isolation must be joint or independent"),
                        },
                    },
                    "score_mode" => options with
                    {
                        ScoreMode = value?.GetValue<string>() switch
                        {
                            "categorical" => ScoreMode.Categorical,
                            "independent_levels" => ScoreMode.IndependentLevels,
                            _ => throw new DecisionSchemaException("score_mode must be categorical or independent_levels"),
                        },
                    },
                    _ => throw new DecisionSchemaException($"unknown option '{key}'"),
                };
            }
            options.Validate();
            return options;
        }
    }

    /// <summary>
    /// A complete decision request, the body of djev's <c>POST /v1/request</c>: a state, named
    /// questions, and options.
    /// </summary>
    public sealed record DecisionRequest
    {
        /// <summary>A string, or structured JSON (an object or an array).</summary>
        public required JsonNode State { get; init; }

        public required QuestionSet Questions { get; init; }

        public DecisionOptions Options { get; init; } = DecisionOptions.Default;

        /// <summary>Carried through to benchmark receipts; never shown to the model.</summary>
        public string? Id { get; init; }

        public static DecisionRequest Create(object state, QuestionSet questions, DecisionOptions? options = null) => new()
        {
            State = PythonJson.ToNode(state) ?? throw new DecisionSchemaException("state is required"),
            Questions = questions,
            Options = options ?? DecisionOptions.Default,
        };

        /// <summary>Parse djev's request body. <c>model</c> is accepted and ignored; <c>images</c> are
        /// refused - this engine reads text.</summary>
        public static DecisionRequest FromJson(string json) => FromJson(JsonNode.Parse(json)
            ?? throw new DecisionSchemaException("the request body is empty"));

        public static DecisionRequest FromJson(JsonNode node)
        {
            if (node is not JsonObject obj) throw new DecisionSchemaException("the request must be a JSON object");
            foreach (var property in obj)
            {
                if (property.Key is not ("state" or "questions" or "model" or "options" or "images"))
                    throw new DecisionSchemaException($"unknown request field '{property.Key}'");
            }
            if (obj["images"] is JsonArray images && images.Count > 0)
                throw new DecisionSchemaException("this inference engine does not support image inputs");
            return new DecisionRequest
            {
                State = obj["state"]?.DeepClone() ?? throw new DecisionSchemaException("state is required"),
                Questions = QuestionSet.FromJson(obj["questions"] ?? throw new DecisionSchemaException("questions are required")),
                Options = DecisionOptions.FromJson(obj["options"]),
            };
        }

        public JsonObject ToJson() => new()
        {
            ["state"] = State.DeepClone(),
            ["questions"] = Questions.ToJson(),
            ["options"] = Options.ToJson(),
        };

        internal bool HasImages() => Questions.Any(q => q.Value.Descriptions().Any(DecisionContracts.HasImage));
    }
}
