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

namespace TensorSharp.Structured
{
    /// <summary>One typed question over a document: what to decide, and the closed set of answers it may
    /// come back with. The options are the language - nothing outside them can be emitted.</summary>
    public sealed class StructuredQuestion
    {
        /// <summary>What the model is being asked, in words.</summary>
        public string Instructions { get; init; } = "";

        /// <summary>The allowed answers. Strings, integers and booleans; at least two, all distinct.</summary>
        public IReadOnlyList<object?> Options { get; init; } = Array.Empty<object?>();

        /// <summary>Optional wording for each option, shown to the model beside the allowed values. A
        /// score question's criteria are what its zero-based indices mean.</summary>
        public IReadOnlyList<string>? Criteria { get; init; }

        /// <summary>Free-form tag carried through to the answer (<c>noul</c>, <c>choice</c>,
        /// <c>score</c>, ...). It does not change decoding - the options do.</summary>
        public string? Kind { get; init; }

        /// <summary>A yes/no question.</summary>
        public static StructuredQuestion Boolean(string instructions) => new()
        {
            Instructions = instructions,
            Options = new object?[] { false, true },
            Kind = "noul",
        };

        /// <summary>A pick-one question over string labels.</summary>
        public static StructuredQuestion Choice(string instructions, params string[] options) => new()
        {
            Instructions = instructions,
            Options = options.Cast<object?>().ToArray(),
            Kind = "choice",
        };

        /// <summary>An ordered scale, answered as a zero-based index into <paramref name="levels"/>.</summary>
        public static StructuredQuestion Score(string instructions, params string[] levels) => new()
        {
            Instructions = instructions,
            Options = Enumerable.Range(0, levels.Length).Cast<object?>().ToArray(),
            Criteria = levels,
            Kind = "score",
        };
    }

    /// <summary>A document and the questions to answer about it. The questions are ordered: that order is
    /// the key order of the JSON object that comes back.</summary>
    public sealed class StructuredRequest
    {
        /// <summary>Identifies this request in a batch and in a benchmark receipt.</summary>
        public string Id { get; init; } = "";

        /// <summary>The text being judged.</summary>
        public string Document { get; init; } = "";

        /// <summary>The questions, in the order their keys appear in the answer.</summary>
        public IReadOnlyDictionary<string, StructuredQuestion> Questions { get; init; } =
            new Dictionary<string, StructuredQuestion>();

        /// <summary>Reject a request before anything is tokenized or a GPU is touched.</summary>
        /// <exception cref="ArgumentException">The request is not answerable as a typed decision.</exception>
        public void Validate()
        {
            if (string.IsNullOrEmpty(Document))
                throw new ArgumentException("A request needs a document to judge.", nameof(Document));
            if (Questions.Count == 0)
                throw new ArgumentException("A request needs at least one question.", nameof(Questions));

            foreach ((string key, StructuredQuestion question) in Questions)
            {
                if (string.IsNullOrEmpty(key))
                    throw new ArgumentException("Every question needs a key.", nameof(Questions));
                if (string.IsNullOrEmpty(question.Instructions))
                    throw new ArgumentException($"Question '{key}' needs instructions.", nameof(Questions));
                if (question.Options.Count < 2)
                    throw new ArgumentException(
                        $"Question '{key}' needs at least two allowed values.", nameof(Questions));
                foreach (object? option in question.Options)
                {
                    if (option is not (string or int or long or bool))
                        throw new ArgumentException(
                            $"Question '{key}': allowed values are strings, integers and booleans.",
                            nameof(Questions));
                }
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (object? option in question.Options)
                {
                    if (!seen.Add(JsonValue.Serialize(option)))
                        throw new ArgumentException(
                            $"Question '{key}' repeats an allowed value.", nameof(Questions));
                }
                if (question.Criteria is { Count: > 0 } criteria
                    && criteria.Count != question.Options.Count)
                    throw new ArgumentException(
                        $"Question '{key}': one criterion per allowed value, or none.", nameof(Questions));
            }
        }
    }

    /// <summary>The answer to one question, with the model's uncalibrated confidence in it.</summary>
    /// <remarks>
    /// <see cref="Probability"/> is a softmax over the allowed tokens of the slot's first discriminating
    /// position, taken from the same diffusion pass. It ranks well and is <b>not</b> a calibrated truth
    /// probability - do not read it as one.
    /// </remarks>
    public sealed class StructuredFieldAnswer
    {
        public required string Key { get; init; }
        public required object? Value { get; init; }
        public required double Probability { get; init; }

        /// <summary>The uncalibrated score of every allowed value, aligned with the question's options.</summary>
        public required IReadOnlyList<double> OptionProbabilities { get; init; }
    }

    /// <summary>A complete typed decision: the JSON document the model settled on, and its fields.</summary>
    public sealed class StructuredPrediction
    {
        public required string Id { get; init; }

        /// <summary>The selected JSON, always a complete member of the request's allowed language.</summary>
        public required string Json { get; init; }

        public required IReadOnlyDictionary<string, StructuredFieldAnswer> Fields { get; init; }

        /// <summary>Canvases this request occupied: more than one when its fields did not fit together.</summary>
        public required int Canvases { get; init; }

        /// <summary>Denoising steps run before the readout.</summary>
        public required int Steps { get; init; }

        /// <summary>The field values alone, ready to serialize or compare.</summary>
        public IReadOnlyDictionary<string, object?> Values =>
            Fields.ToDictionary(x => x.Key, x => x.Value.Value);
    }

    /// <summary>JSON for the option values, in the one spelling the canvas and the answer agree on.</summary>
    internal static class JsonValue
    {
        private static readonly JsonSerializerOptions Compact = new() { WriteIndented = false };

        public static string Serialize(object? value) => JsonSerializer.Serialize(value, Compact);

        public static object? Parse(JsonNode? node) => node switch
        {
            null => null,
            _ => node.GetValueKind() switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Number => node.GetValue<long>() is var n && n >= int.MinValue && n <= int.MaxValue
                    ? (int)n : n,
                JsonValueKind.String => node.GetValue<string>(),
                _ => throw new FormatException($"Unsupported JSON value kind {node.GetValueKind()}."),
            },
        };
    }
}
