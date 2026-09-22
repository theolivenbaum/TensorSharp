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
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Decisions
{
    using JsonValue = System.Text.Json.Nodes.JsonValue;

    /// <summary>
    /// djev's request limits and its probability-to-answer conversion (<c>djev/contracts.py</c>), with
    /// the same bounds and the same arithmetic.
    /// </summary>
    public static class DecisionContracts
    {
        public const int MaxQuestions = 32;
        public const int MaxChoiceOptions = 255;
        public const int MaxScoreLevels = 10;
        public const int MaxStateCharacters = 20_000;
        public const int MaxInstructionsCharacters = 2_000;
        public const int MaxCriterionCharacters = 500;

        /// <summary>Reject a request djev would reject, with djev's reasons, before any tokenization.</summary>
        public static void Validate(DecisionRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            request.Options.Validate();
            if (request.State.GetValueKind() is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
                throw new DecisionSchemaException("state must be a string, an object or an array");
            if (CharacterCount(request.State, imageDescriptions: false) > MaxStateCharacters)
                throw new DecisionSchemaException($"state exceeds {MaxStateCharacters} Unicode characters");
            if (request.Questions.Count is < 1 or > MaxQuestions)
                throw new DecisionSchemaException($"a request needs between 1 and {MaxQuestions} questions");
            if (request.HasImages())
                throw new DecisionSchemaException("this inference engine does not support image inputs");

            foreach ((string _, Question question) in request.Questions)
            {
                switch (question.Type)
                {
                    case QuestionType.Choice when question.Options.Count is < 1 or > MaxChoiceOptions:
                        throw new DecisionSchemaException($"choice questions need between 1 and {MaxChoiceOptions} options");
                    case QuestionType.Score when question.Levels.Count is < 2 or > MaxScoreLevels:
                        throw new DecisionSchemaException($"score questions need between 2 and {MaxScoreLevels} levels");
                }
                if (question.Type == QuestionType.Choice
                    && question.Options.Select(o => o.Key).Distinct(StringComparer.Ordinal).Count() != question.Options.Count)
                    throw new DecisionSchemaException("choice option names must be unique");

                if (CharacterCount(question.Instructions) > MaxInstructionsCharacters)
                    throw new DecisionSchemaException($"instructions exceed {MaxInstructionsCharacters} Unicode characters");
                IEnumerable<JsonNode?> criteria = question.Type switch
                {
                    QuestionType.Noul => question.HasNoulCriteria
                        ? new[] { question.TrueCriterion, question.FalseCriterion } : Array.Empty<JsonNode?>(),
                    QuestionType.Choice => question.Options.Select(o => (JsonNode?)JsonValue.Create(o.Key))
                        .Concat(question.Options.Select(o => o.Value)),
                    _ => question.Levels,
                };
                if (criteria.Any(c => CharacterCount(c) > MaxCriterionCharacters))
                    throw new DecisionSchemaException(
                        $"criterion descriptions and choice names cannot exceed {MaxCriterionCharacters} Unicode characters");
            }
        }

        /// <summary>Unicode code points; structured values count their compact, key-sorted JSON.</summary>
        public static int CharacterCount(JsonNode? value, bool imageDescriptions = true)
        {
            if (value is null) return 0;
            if (value.GetValueKind() == JsonValueKind.String) return CodePoints(value.GetValue<string>());
            return CodePoints(PythonJson.Dumps(value, sortKeys: true));
        }

        private static int CodePoints(string text)
        {
            int count = 0;
            foreach (var _ in text.EnumerateRunes()) count++;
            return count;
        }

        /// <summary>djev's opt-in image description, <c>{"image": "data:..."}</c>, anywhere in a value.</summary>
        internal static bool HasImage(JsonNode? value) => value switch
        {
            JsonObject obj when obj["image"] is JsonValue image && image.GetValueKind() == JsonValueKind.String
                                && image.GetValue<string>().StartsWith("data:", StringComparison.Ordinal) => true,
            JsonObject obj => obj.Any(p => HasImage(p.Value)),
            JsonArray array => array.Any(HasImage),
            _ => false,
        };

        /// <summary>
        /// A stable softmax over a closed label set that keeps impossible (-∞) labels at zero.
        ///
        /// Missing or invalid evidence fails rather than being papered over with a uniform distribution:
        /// NaN and +∞ are invalid backend output, and a set with no finite member has no evidence at all.
        /// </summary>
        public static double[] NormalizeLogprobs(IReadOnlyList<double> logprobs)
        {
            if (logprobs.Count == 0) throw new DecisionBackendException("at least one log probability is required");
            if (logprobs.Any(v => double.IsNaN(v) || double.IsPositiveInfinity(v)))
                throw new DecisionBackendException("log probabilities cannot contain NaN or positive infinity");
            double peak = logprobs.Max();
            if (double.IsNegativeInfinity(peak))
                throw new DecisionBackendException("at least one log probability must be finite");
            var weights = logprobs.Select(v => Math.Exp(v - peak)).ToArray();
            double total = Fsum(weights);
            return weights.Select(w => w / total).ToArray();
        }

        private static double[] Validated(IReadOnlyList<double> probabilities, int count)
        {
            if (probabilities.Count != count)
                throw new DecisionBackendException($"expected {count} probabilities, received {probabilities.Count}");
            if (probabilities.Any(v => !double.IsFinite(v) || v < 0 || v > 1))
                throw new DecisionBackendException("probabilities must be finite numbers between 0 and 1");
            double total = Fsum(probabilities);
            if (Math.Abs(total - 1.0) > Math.Max(1e-6 * Math.Max(Math.Abs(total), 1.0), 1e-8))
                throw new DecisionBackendException("probabilities must sum to 1");
            return probabilities.Select(v => v / total).ToArray();
        }

        /// <summary><c>1 - H(p) / log(n)</c>, clipped to [0, 1]; a single option is fully concentrated.</summary>
        public static double EntropyConcentration(IReadOnlyList<double> probabilities)
        {
            if (probabilities.Count == 1) return 1.0;
            double entropy = -Fsum(probabilities.Where(p => p > 0).Select(p => p * Math.Log(p)).ToArray());
            return Math.Clamp(1.0 - entropy / Math.Log(probabilities.Count), 0.0, 1.0);
        }

        /// <summary>
        /// A typed answer from a complete, normalised distribution. Noul takes <c>[false, true]</c>;
        /// Choice follows option order and breaks ties towards the first; Score follows level order and
        /// reports the expected index.
        /// </summary>
        public static Answer AnswerFrom(Question question, IReadOnlyList<double> probabilities)
        {
            switch (question.Type)
            {
                case QuestionType.Noul:
                {
                    double[] values = Validated(probabilities, 2);
                    return new Answer { Type = QuestionType.Noul, Noul = values[1] };
                }
                case QuestionType.Choice:
                {
                    var labels = question.Options.Select(o => o.Key).ToList();
                    double[] values = Validated(probabilities, labels.Count);
                    int winner = 0;
                    for (int i = 1; i < values.Length; i++) if (values[i] > values[winner]) winner = i;
                    return new Answer
                    {
                        Type = QuestionType.Choice,
                        Choice = labels[winner],
                        Probabilities = labels.Select((l, i) => new KeyValuePair<string, double>(l, values[i])).ToList(),
                        Confidence = EntropyConcentration(values),
                    };
                }
                default:
                {
                    double[] values = Validated(probabilities, question.Levels.Count);
                    return new Answer
                    {
                        Type = QuestionType.Score,
                        Score = Fsum(values.Select((v, i) => i * v).ToArray()),
                        Legend = question.Levels.Select((l, i) => new KeyValuePair<string, JsonNode?>(
                            i.ToString(CultureInfo.InvariantCulture), l?.DeepClone())).ToList(),
                        Probabilities = values.Select((v, i) => new KeyValuePair<string, double>(
                            i.ToString(CultureInfo.InvariantCulture), v)).ToList(),
                        Confidence = EntropyConcentration(values),
                    };
                }
            }
        }

        /// <summary>Python's <c>math.fsum</c>: an exactly rounded sum (Shewchuk's partials).</summary>
        public static double Fsum(IReadOnlyList<double> values)
        {
            var partials = new List<double>();
            foreach (double value in values)
            {
                double x = value;
                int i = 0;
                for (int j = 0; j < partials.Count; j++)
                {
                    double y = partials[j];
                    if (Math.Abs(x) < Math.Abs(y)) (x, y) = (y, x);
                    double hi = x + y;
                    double lo = y - (hi - x);
                    if (lo != 0.0) partials[i++] = lo;
                    x = hi;
                }
                partials.RemoveRange(i, partials.Count - i);
                partials.Add(x);
            }
            if (partials.Count == 0) return 0.0;
            // Round-half-even correction across the top two partials, as CPython does.
            int n = partials.Count - 1;
            double total = partials[n];
            while (n > 0)
            {
                double x = total;
                double y = partials[--n];
                total = x + y;
                double yr = total - x;
                double lo = y - yr;
                if (lo != 0.0)
                {
                    if (n > 0 && ((lo < 0 && partials[n - 1] < 0) || (lo > 0 && partials[n - 1] > 0)))
                    {
                        double twice = lo * 2;
                        double candidate = total + twice;
                        if (twice == candidate - total) total = candidate;
                    }
                    break;
                }
            }
            return total;
        }

        /// <summary>Entropy of a distribution, in nats.</summary>
        internal static double Entropy(IReadOnlyList<double> probabilities) =>
            -Fsum(probabilities.Where(p => p > 0).Select(p => p * Math.Log(p)).ToArray());
    }
}
