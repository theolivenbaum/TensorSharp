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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>The three decision shapes djev (and Jev) define.</summary>
    public enum QuestionType
    {
        /// <summary>Yes/no: the answer is the probability the statement holds.</summary>
        Noul,

        /// <summary>One of the caller's named options.</summary>
        Choice,

        /// <summary>A position on an ordered rubric, answered as the expected zero-based level.</summary>
        Score,
    }

    /// <summary>
    /// One typed question. Instructions and criteria descriptions are text, structured JSON, or nothing;
    /// structured values reach the model as compact JSON, exactly as djev writes them.
    /// </summary>
    public sealed class Question
    {
        public QuestionType Type { get; }

        /// <summary>What to decide. A string, a JSON value, or null.</summary>
        public JsonNode? Instructions { get; }

        /// <summary>Noul only: what <c>true</c> means. Null when the question carries no criteria.</summary>
        public JsonNode? TrueCriterion { get; }

        /// <summary>Noul only: what <c>false</c> means.</summary>
        public JsonNode? FalseCriterion { get; }

        /// <summary>Noul only: whether a <c>criteria</c> object was given at all (it may hold two nulls).</summary>
        public bool HasNoulCriteria { get; }

        /// <summary>Choice only: the options in the caller's order, each with an optional description.
        /// Option order is significant: it decides which option gets which answer label.</summary>
        public IReadOnlyList<KeyValuePair<string, JsonNode?>> Options { get; }

        /// <summary>Score only: the rubric levels, lowest first.</summary>
        public IReadOnlyList<JsonNode?> Levels { get; }

        private Question(QuestionType type, JsonNode? instructions,
            IReadOnlyList<KeyValuePair<string, JsonNode?>>? options = null,
            IReadOnlyList<JsonNode?>? levels = null,
            bool hasNoulCriteria = false, JsonNode? trueCriterion = null, JsonNode? falseCriterion = null)
        {
            Type = type;
            Instructions = instructions;
            Options = options ?? Array.Empty<KeyValuePair<string, JsonNode?>>();
            Levels = levels ?? Array.Empty<JsonNode?>();
            HasNoulCriteria = hasNoulCriteria;
            TrueCriterion = trueCriterion;
            FalseCriterion = falseCriterion;
        }

        /// <summary>A yes/no question. With neither criterion, the labels describe themselves.</summary>
        public static Question Noul(object? instructions, object? trueCriterion = null, object? falseCriterion = null)
        {
            bool has = trueCriterion is not null || falseCriterion is not null;
            return new Question(QuestionType.Noul, PythonJson.ToNode(instructions), hasNoulCriteria: has,
                trueCriterion: PythonJson.ToNode(trueCriterion), falseCriterion: PythonJson.ToNode(falseCriterion));
        }

        /// <summary>A pick-one question over bare option names.</summary>
        public static Question Choice(object? instructions, params string[] options) =>
            Choice(instructions, options.Select(o => new KeyValuePair<string, object?>(o, null)));

        /// <summary>A pick-one question over named, described options.</summary>
        public static Question Choice(object? instructions, params (string Option, object? Description)[] options) =>
            Choice(instructions, options.Select(o => new KeyValuePair<string, object?>(o.Option, o.Description)));

        /// <summary>A pick-one question; the enumeration order is the option order.</summary>
        public static Question Choice(object? instructions, IEnumerable<KeyValuePair<string, object?>> options) =>
            new(QuestionType.Choice, PythonJson.ToNode(instructions),
                options: options.Select(o => new KeyValuePair<string, JsonNode?>(o.Key, PythonJson.ToNode(o.Value))).ToList());

        /// <summary>An ordered rubric, lowest level first.</summary>
        public static Question Score(object? instructions, params object?[] levels) =>
            new(QuestionType.Score, PythonJson.ToNode(instructions), levels: levels.Select(PythonJson.ToNode).ToList());

        /// <summary>
        /// Read a question in djev's wire format: <c>{"type": "noul"|"choice"|"score", "instructions": ...,
        /// "criteria": ...}</c>, where Noul criteria are <c>{"true", "false"}</c>, Choice criteria an object of
        /// option → description, and Score criteria an array of levels.
        /// </summary>
        public static Question FromJson(JsonNode node)
        {
            if (node is not JsonObject obj) throw new DecisionSchemaException("a question must be a JSON object");
            string type = obj["type"]?.GetValue<string>() ?? "noul";
            JsonNode? instructions = obj["instructions"]?.DeepClone();
            JsonNode? criteria = obj["criteria"];
            foreach (var property in obj)
            {
                if (property.Key is not ("type" or "instructions" or "criteria"))
                    throw new DecisionSchemaException($"unknown question field '{property.Key}'");
            }
            switch (type)
            {
                case "noul":
                    if (criteria is null) return new Question(QuestionType.Noul, instructions);
                    if (criteria is not JsonObject noul || noul.Any(p => p.Key is not ("true" or "false")))
                        throw new DecisionSchemaException("noul criteria are an object with 'true' and 'false'");
                    return new Question(QuestionType.Noul, instructions, hasNoulCriteria: true,
                        trueCriterion: noul["true"]?.DeepClone(), falseCriterion: noul["false"]?.DeepClone());
                case "choice":
                    if (criteria is not JsonObject named)
                        throw new DecisionSchemaException("choice criteria are an object of option names");
                    return new Question(QuestionType.Choice, instructions,
                        options: named.Select(p => new KeyValuePair<string, JsonNode?>(p.Key, p.Value?.DeepClone())).ToList());
                case "score":
                    if (criteria is not JsonArray levels)
                        throw new DecisionSchemaException("score criteria are an array of levels");
                    return new Question(QuestionType.Score, instructions,
                        levels: levels.Select(l => l?.DeepClone()).ToList());
                default:
                    throw new DecisionSchemaException($"unsupported question type '{type}'");
            }
        }

        /// <summary>The question in djev's wire format, fields in pydantic's order (type, instructions,
        /// criteria). This is also the identity djev hashes, so the order matters.</summary>
        public JsonObject ToJson()
        {
            var result = new JsonObject
            {
                ["type"] = TypeName(Type),
                ["instructions"] = Instructions?.DeepClone(),
            };
            result["criteria"] = Type switch
            {
                QuestionType.Noul => HasNoulCriteria
                    ? new JsonObject { ["true"] = TrueCriterion?.DeepClone(), ["false"] = FalseCriterion?.DeepClone() }
                    : null,
                QuestionType.Choice => new JsonObject(Options.Select(
                    o => new KeyValuePair<string, JsonNode?>(o.Key, o.Value?.DeepClone()))),
                _ => new JsonArray(Levels.Select(l => l?.DeepClone()).ToArray()),
            };
            return result;
        }

        /// <summary>The answer labels the caller sees, in distribution order: <c>no, yes</c> for Noul, the
        /// option names for Choice, <c>"0".."n-1"</c> for Score.</summary>
        public IReadOnlyList<string> Labels => Type switch
        {
            QuestionType.Noul => new[] { "no", "yes" },
            QuestionType.Choice => Options.Select(o => o.Key).ToList(),
            _ => Enumerable.Range(0, Levels.Count).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToList(),
        };

        /// <summary>Every description the question carries: instructions first, then the criteria (Noul
        /// criteria in <c>false, true</c> order).</summary>
        internal IEnumerable<JsonNode?> Descriptions()
        {
            yield return Instructions;
            switch (Type)
            {
                case QuestionType.Noul:
                    if (HasNoulCriteria) { yield return FalseCriterion; yield return TrueCriterion; }
                    break;
                case QuestionType.Choice:
                    foreach (var option in Options) yield return option.Value;
                    break;
                default:
                    foreach (var level in Levels) yield return level;
                    break;
            }
        }

        public static string TypeName(QuestionType type) => type switch
        {
            QuestionType.Noul => "noul",
            QuestionType.Choice => "choice",
            _ => "score",
        };
    }

    /// <summary>
    /// Named questions in order. The names are the caller's bookkeeping only: they never reach the model,
    /// so renaming a question cannot change its answer. The order can - in a joint read, question
    /// <c>i</c> is "Question i" on the canvas.
    /// </summary>
    public sealed class QuestionSet : IEnumerable<KeyValuePair<string, Question>>
    {
        private readonly List<string> _order = new();
        private readonly Dictionary<string, Question> _questions = new(StringComparer.Ordinal);

        public QuestionSet() { }

        public QuestionSet(IEnumerable<KeyValuePair<string, Question>> questions)
        {
            foreach ((string id, Question question) in questions) Add(id, question);
        }

        public int Count => _order.Count;

        public IReadOnlyList<string> Ids => _order;

        public Question this[string id] => _questions[id];

        public QuestionSet Add(string id, Question question)
        {
            ArgumentNullException.ThrowIfNull(id);
            ArgumentNullException.ThrowIfNull(question);
            if (!_questions.TryAdd(id, question))
                throw new ArgumentException($"question id '{id}' is already in the set", nameof(id));
            _order.Add(id);
            return this;
        }

        public bool Contains(string id) => _questions.ContainsKey(id);

        /// <summary>Questions keyed by id in djev's wire format.</summary>
        public static QuestionSet FromJson(JsonNode node)
        {
            if (node is not JsonObject obj) throw new DecisionSchemaException("questions must be a JSON object");
            var set = new QuestionSet();
            foreach ((string id, JsonNode? question) in obj)
                set.Add(id, Question.FromJson(question ?? throw new DecisionSchemaException($"question '{id}' is null")));
            return set;
        }

        public JsonObject ToJson() =>
            new(_order.Select(id => new KeyValuePair<string, JsonNode?>(id, _questions[id].ToJson())));

        public IEnumerator<KeyValuePair<string, Question>> GetEnumerator()
        {
            foreach (string id in _order) yield return new KeyValuePair<string, Question>(id, _questions[id]);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
