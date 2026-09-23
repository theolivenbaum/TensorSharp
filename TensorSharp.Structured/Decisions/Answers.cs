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
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>A request is not answerable: it breaks a limit, or cannot compile into the canvas. djev
    /// answers these with HTTP 422, before any model work.</summary>
    public class DecisionSchemaException : ArgumentException
    {
        public DecisionSchemaException(string message) : base(message) { }
    }

    /// <summary>The model returned incomplete or invalid evidence. djev answers these with HTTP 502; the
    /// answer is never fabricated from what did come back.</summary>
    public class DecisionBackendException : InvalidOperationException
    {
        public DecisionBackendException(string message) : base(message) { }
    }

    /// <summary>
    /// One typed answer, shaped like djev's.
    ///
    /// <see cref="Confidence"/> is normalised entropy concentration, <c>1 - H(p) / log(n)</c>: how peaked
    /// the distribution is, not how often such answers are right. It is not calibrated.
    /// </summary>
    public sealed record Answer
    {
        public required QuestionType Type { get; init; }

        /// <summary>Noul: the probability the statement holds, relative to the two allowed labels.</summary>
        public double? Noul { get; init; }

        /// <summary>Choice: the most probable option (the first one on a tie).</summary>
        public string? Choice { get; init; }

        /// <summary>Score: the expected zero-based level, <c>Σ i·p(i)</c> - not the most probable one.</summary>
        public double? Score { get; init; }

        /// <summary>Choice and Score: the distribution, in option / level order. Null for Noul.</summary>
        public IReadOnlyList<KeyValuePair<string, double>>? Probabilities { get; init; }

        /// <summary>Score: what each level index means.</summary>
        public IReadOnlyList<KeyValuePair<string, JsonNode?>>? Legend { get; init; }

        /// <summary>Choice and Score: entropy concentration of <see cref="Probabilities"/>. Null for Noul,
        /// as in djev.</summary>
        public double? Confidence { get; init; }

        /// <summary>The distribution as <c>label → p</c> for every type; Noul reports <c>no</c> and
        /// <c>yes</c>.</summary>
        public IReadOnlyDictionary<string, double> Distribution => Type == QuestionType.Noul
            ? new Dictionary<string, double> { ["no"] = 1 - Noul!.Value, ["yes"] = Noul!.Value }
            : Probabilities!.ToDictionary(p => p.Key, p => p.Value);

        /// <summary>The probability of one label; Noul accepts <c>yes</c>/<c>true</c> and
        /// <c>no</c>/<c>false</c>.</summary>
        public double ProbabilityOf(string label)
        {
            if (Type == QuestionType.Noul)
            {
                return label switch
                {
                    "yes" or "true" => Noul!.Value,
                    "no" or "false" => 1 - Noul!.Value,
                    _ => throw new KeyNotFoundException($"'{label}' is not a noul label"),
                };
            }
            foreach ((string key, double p) in Probabilities!)
                if (key == label) return p;
            throw new KeyNotFoundException($"'{label}' is not an option of this answer");
        }

        /// <summary>The most probable label: the option for Choice, the level index for Score, and
        /// <c>yes</c> or <c>no</c> for Noul.</summary>
        public string Label => Type switch
        {
            QuestionType.Noul => Noul >= 0.5 ? "yes" : "no",
            QuestionType.Choice => Choice!,
            _ => Probabilities!.Select((p, i) => (p.Value, i)).MaxBy(x => x.Value).i.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };

        /// <summary>The answer in djev's wire format.</summary>
        public JsonObject ToJson()
        {
            var result = new JsonObject { ["type"] = Question.TypeName(Type) };
            switch (Type)
            {
                case QuestionType.Noul:
                    result["noul"] = Noul;
                    break;
                case QuestionType.Choice:
                    result["choice"] = Choice;
                    result["probabilities"] = Map(Probabilities!);
                    result["confidence"] = Confidence;
                    break;
                default:
                    result["score"] = Score;
                    result["legend"] = new JsonObject(Legend!.Select(
                        l => new KeyValuePair<string, JsonNode?>(l.Key, l.Value?.DeepClone())));
                    result["probabilities"] = Map(Probabilities!);
                    result["confidence"] = Confidence;
                    break;
            }
            return result;
        }

        private static JsonObject Map(IEnumerable<KeyValuePair<string, double>> values) =>
            new(values.Select(v => new KeyValuePair<string, JsonNode?>(v.Key, v.Value)));
    }

    /// <summary>Tokens the reads physically consumed. Samples and independently read questions each
    /// count their prompt again - this is work done, not unique text.</summary>
    public sealed record DecisionUsage(int InputTokens, int OutputTokens);

    /// <summary>The answer to a whole question set.</summary>
    public sealed record DecisionResult
    {
        public required string Model { get; init; }

        /// <summary>Answers in question order.</summary>
        public required IReadOnlyList<KeyValuePair<string, Answer>> Answers { get; init; }

        public required DecisionUsage Usage { get; init; }

        /// <summary>djev's diagnostics block (probability basis, label mass, read counts, canvas width),
        /// present when <see cref="DecisionOptions.Diagnostics"/> asked for it.</summary>
        public JsonObject? Diagnostics { get; init; }

        /// <summary>Validating, compiling and tokenizing, before any model work.</summary>
        public double CompileMs { get; init; }

        /// <summary>Waiting on the reads. Shared by every request of a batch that ran together.</summary>
        public double ModelMs { get; init; }

        /// <summary>Canvas widths the reads ran at, one per distinct compiled schema.</summary>
        public IReadOnlyList<int> CanvasWidths { get; init; } = Array.Empty<int>();

        public Answer this[string id]
        {
            get
            {
                foreach ((string key, Answer answer) in Answers)
                    if (key == id) return answer;
                throw new KeyNotFoundException($"no answer for question '{id}'");
            }
        }

        public bool TryGet(string id, out Answer answer)
        {
            foreach ((string key, Answer value) in Answers)
            {
                if (key == id) { answer = value; return true; }
            }
            answer = null!;
            return false;
        }

        /// <summary>The response body djev's <c>POST /v1/request</c> returns.</summary>
        public JsonObject ToJson()
        {
            var body = new JsonObject
            {
                ["model"] = Model,
                ["answers"] = new JsonObject(Answers.Select(a => new KeyValuePair<string, JsonNode?>(a.Key, a.Value.ToJson()))),
                ["usage"] = new JsonObject { ["input_tokens"] = Usage.InputTokens, ["output_tokens"] = Usage.OutputTokens },
            };
            if (Diagnostics is not null) body["diagnostics"] = Diagnostics.DeepClone();
            return body;
        }

        public string ToJsonString(bool indented = false) =>
            ToJson().ToJsonString(new JsonSerializerOptions { WriteIndented = indented });
    }
}
