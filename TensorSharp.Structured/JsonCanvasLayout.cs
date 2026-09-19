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
using System.Text.Json.Nodes;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Structured
{
    /// <summary>
    /// A JSON canvas whose scaffolding is fixed and whose answer slots are free.
    ///
    /// Every allowed answer is tokenized as a complete JSON document, so the token positions every
    /// candidate agrees on - the braces, the quoted keys, the separators - are the same everywhere and can
    /// be pinned for the whole denoise. What is left free is exactly the positions where candidates differ.
    /// Those denoise unrestricted; nothing constrains them until the readout, which then picks the highest
    /// scoring <em>allowed</em> token at each, left to right, narrowing to a single candidate per field.
    ///
    /// Candidates of a field are padded to a common width with a single whitespace token, so a field
    /// occupies the same span whichever answer wins, and the fields compose without enumerating their
    /// Cartesian product.
    /// </summary>
    public sealed class JsonCanvasLayout
    {
        /// <summary>One answer slot: where the field's span starts, its padded candidates, and the
        /// positions within the span where those candidates disagree.</summary>
        private sealed record Slot(
            string Key,
            int Offset,
            int[][] Candidates,
            IReadOnlyList<(int Position, int Column, int[] Allowed)> Variables);

        private readonly ITokenizer _tokenizer;
        private readonly Slot[] _slots;
        private readonly int _templateLength;

        /// <summary>The questions this canvas answers, in key order.</summary>
        public IReadOnlyDictionary<string, StructuredQuestion> Questions { get; }

        /// <summary>The canvas the model denoises: the JSON template, then end-of-sequence padding.</summary>
        public int[] SeedCanvas { get; }

        /// <summary>Which canvas positions are held at their seed value for the whole denoise - every
        /// position except the answer slots' variable ones.</summary>
        public bool[] PinnedPositions { get; }

        /// <summary>Per canvas position, the tokens whose score the readout needs; null where none.</summary>
        public int[][] LogprobTokenIds { get; }

        /// <summary>Canvas positions that denoise freely.</summary>
        public IReadOnlyList<int> VariablePositions { get; }

        private JsonCanvasLayout(
            ITokenizer tokenizer,
            IReadOnlyDictionary<string, StructuredQuestion> questions,
            Slot[] slots,
            int[] seedCanvas,
            bool[] pinned,
            int[][] logprobTokenIds,
            int templateLength)
        {
            _tokenizer = tokenizer;
            _slots = slots;
            _templateLength = templateLength;
            Questions = questions;
            SeedCanvas = seedCanvas;
            PinnedPositions = pinned;
            LogprobTokenIds = logprobTokenIds;
            VariablePositions = slots.SelectMany(s => s.Variables.Select(v => v.Position)).ToArray();
        }

        /// <summary>The padded token run each allowed value of <paramref name="key"/> occupies, aligned
        /// with that question's options. Every run is the same length, so a field holds the same span
        /// whichever answer wins.</summary>
        public IReadOnlyList<int[]> CandidateTokens(string key) => SlotFor(key).Candidates;

        /// <summary>Where <paramref name="key"/>'s answer span starts on the canvas.</summary>
        public int SlotOffset(string key) => SlotFor(key).Offset;

        private Slot SlotFor(string key) =>
            _slots.FirstOrDefault(s => s.Key == key)
            ?? throw new KeyNotFoundException($"This canvas does not carry question '{key}'.");

        /// <summary>
        /// Compile the questions into a canvas of <paramref name="canvasLength"/> positions.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// The questions do not fit the canvas, or the tokenizer does not round-trip them - the canvas is
        /// only sound if concatenating the token runs really does decode to the intended JSON, which token
        /// boundaries do not guarantee, so it is checked rather than assumed.
        /// </exception>
        public static JsonCanvasLayout Compile(
            ITokenizer tokenizer,
            IReadOnlyDictionary<string, StructuredQuestion> questions,
            int canvasLength,
            int eosTokenId)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ArgumentNullException.ThrowIfNull(questions);
            if (questions.Count == 0)
                throw new ArgumentException("A canvas needs at least one question.", nameof(questions));

            List<int> space = tokenizer.Encode(" ", addSpecial: false);
            if (space.Count != 1 || tokenizer.Decode(space) != " ")
                throw new ArgumentException(
                    "This tokenizer has no single whitespace token to pad answer slots with.",
                    nameof(tokenizer));
            int pad = space[0];

            var template = new List<int>(tokenizer.Encode("{\n", addSpecial: false));
            var slots = new List<Slot>(questions.Count);
            bool first = true;
            foreach ((string key, StructuredQuestion question) in questions)
            {
                if (!first) template.AddRange(tokenizer.Encode(",\n", addSpecial: false));
                first = false;

                int[][] candidates = question.Options
                    .Select(v => tokenizer
                        .Encode(JsonValue.Serialize(key) + ": " + JsonValue.Serialize(v), addSpecial: false)
                        .ToArray())
                    .ToArray();
                int span = candidates.Max(c => c.Length);
                candidates = candidates
                    .Select(c => c.Concat(Enumerable.Repeat(pad, span - c.Length)).ToArray())
                    .ToArray();

                int offset = template.Count;
                var variables = new List<(int, int, int[])>();
                for (int column = 0; column < span; column++)
                {
                    int[] allowed = candidates.Select(c => c[column]).Distinct().Order().ToArray();
                    if (allowed.Length > 1) variables.Add((offset + column, column, allowed));
                }
                slots.Add(new Slot(key, offset, candidates, variables));
                template.AddRange(candidates[0]);
            }
            template.AddRange(tokenizer.Encode("\n}", addSpecial: false));

            if (template.Count > canvasLength)
                throw new ArgumentException(
                    $"These {questions.Count} question(s) need {template.Count} canvas positions; " +
                    $"the canvas holds {canvasLength}.", nameof(questions));

            VerifyRoundTrip(tokenizer, questions, template, slots);

            int[] seed = template
                .Concat(Enumerable.Repeat(eosTokenId, canvasLength - template.Count))
                .ToArray();
            var pinned = new bool[canvasLength];
            Array.Fill(pinned, true);
            var logprobIds = new int[canvasLength][];
            foreach (Slot slot in slots)
            {
                foreach ((int position, _, int[] allowed) in slot.Variables)
                {
                    pinned[position] = false;
                    logprobIds[position] = allowed;
                }
            }
            return new JsonCanvasLayout(
                tokenizer, questions, slots.ToArray(), seed, pinned, logprobIds, template.Count);
        }

        /// <summary>The canvas is only sound if the concatenated token runs decode to the intended JSON.
        /// Token boundaries do not guarantee that, so every candidate substitution is checked once, at
        /// compile time, rather than discovered as an unparseable answer later.</summary>
        private static void VerifyRoundTrip(
            ITokenizer tokenizer,
            IReadOnlyDictionary<string, StructuredQuestion> questions,
            List<int> template,
            List<Slot> slots)
        {
            var expected = questions.ToDictionary(x => x.Key, x => x.Value.Options[0]);
            if (!Matches(tokenizer, template, expected))
                throw new ArgumentException(
                    "The canvas template does not decode to the JSON it was built from.");

            foreach (Slot slot in slots)
            {
                StructuredQuestion question = questions[slot.Key];
                for (int i = 0; i < slot.Candidates.Length; i++)
                {
                    var ids = new List<int>(template);
                    for (int j = 0; j < slot.Candidates[i].Length; j++)
                        ids[slot.Offset + j] = slot.Candidates[i][j];
                    var want = new Dictionary<string, object?>(expected) { [slot.Key] = question.Options[i] };
                    if (!Matches(tokenizer, ids, want))
                        throw new ArgumentException(
                            $"Question '{slot.Key}' does not tokenize consistently for value " +
                            $"{JsonValue.Serialize(question.Options[i])}.");
                }
            }
        }

        private static bool Matches(ITokenizer tokenizer, List<int> ids, Dictionary<string, object?> expected)
        {
            try
            {
                if (JsonNode.Parse(tokenizer.Decode(ids)) is not JsonObject parsed) return false;
                if (!parsed.Select(x => x.Key).SequenceEqual(expected.Keys)) return false;
                return expected.All(kv =>
                    JsonValue.Serialize(JsonValue.Parse(parsed[kv.Key])) == JsonValue.Serialize(kv.Value));
            }
            catch (Exception ex) when (ex is FormatException or System.Text.Json.JsonException)
            {
                return false;
            }
        }

        /// <summary>
        /// Choose one complete allowed JSON document from the read's per-position scores.
        ///
        /// Per field the readout walks its variable positions left to right, keeping the candidates that
        /// carry the highest-scoring allowed token at each. The first position where candidates disagree
        /// does most of the work; later ones resolve what survives, and a unique suffix is forced. Later
        /// decisions use the scores of the same diffusion pass, not likelihoods conditioned on the prefix
        /// just chosen.
        /// </summary>
        /// <exception cref="ArgumentException">The read did not report a position the readout needs.</exception>
        public StructuredPrediction Select(
            string id, IReadOnlyList<DiffusionPositionLogprobs> logprobs, int steps, int canvases)
        {
            ArgumentNullException.ThrowIfNull(logprobs);
            var ids = new List<int>(SeedCanvas[.._templateLength]);
            var fields = new Dictionary<string, StructuredFieldAnswer>(_slots.Length);

            foreach (Slot slot in _slots)
            {
                var active = new bool[slot.Candidates.Length];
                Array.Fill(active, true);
                foreach ((int position, int column, int[] allowed) in slot.Variables)
                {
                    float[] scores = ScoresAt(logprobs, position, allowed);
                    int best = -1;
                    for (int c = 0; c < slot.Candidates.Length; c++)
                    {
                        if (!active[c]) continue;
                        int rank = Array.IndexOf(allowed, slot.Candidates[c][column]);
                        if (best < 0 || scores[rank] > scores[Array.IndexOf(allowed, slot.Candidates[best][column])])
                            best = c;
                    }
                    int chosen = slot.Candidates[best][column];
                    for (int c = 0; c < slot.Candidates.Length; c++)
                        active[c] &= slot.Candidates[c][column] == chosen;
                }

                int selected = Array.IndexOf(active, true);
                for (int j = 0; j < slot.Candidates[selected].Length; j++)
                    ids[slot.Offset + j] = slot.Candidates[selected][j];

                IReadOnlyList<double> optionProbabilities = OptionProbabilities(slot, logprobs);
                fields[slot.Key] = new StructuredFieldAnswer
                {
                    Key = slot.Key,
                    Value = Questions[slot.Key].Options[selected],
                    Probability = optionProbabilities[selected],
                    OptionProbabilities = optionProbabilities,
                };
            }

            return new StructuredPrediction
            {
                Id = id,
                Json = Decode(ids, fields),
                Fields = fields,
                Steps = steps,
                Canvases = canvases,
            };
        }

        /// <summary>
        /// An uncalibrated score per allowed value: the softmax over the allowed tokens of the field's
        /// first discriminating position. Values that share a token there share its mass - the number
        /// separates them no better than that position does - and a field with no free position at all
        /// (a single candidate) is reported as certain.
        /// </summary>
        private static IReadOnlyList<double> OptionProbabilities(
            Slot slot, IReadOnlyList<DiffusionPositionLogprobs> logprobs)
        {
            int n = slot.Candidates.Length;
            if (slot.Variables.Count == 0) return Enumerable.Repeat(1.0 / n, n).ToArray();

            (int position, int column, int[] allowed) = slot.Variables[0];
            float[] scores = ScoresAt(logprobs, position, allowed);
            double max = scores.Max();
            double[] mass = scores.Select(s => Math.Exp(s - max)).ToArray();
            double total = mass.Sum();
            return slot.Candidates
                .Select(c => mass[Array.IndexOf(allowed, c[column])] / total)
                .ToArray();
        }

        /// <summary>The read's scores for <paramref name="allowed"/> at <paramref name="position"/>,
        /// in that order. The read is asked for exactly these ids, so this is a lookup, not a search.</summary>
        private static float[] ScoresAt(
            IReadOnlyList<DiffusionPositionLogprobs> logprobs, int position, int[] allowed)
        {
            if (position >= logprobs.Count)
                throw new ArgumentException(
                    $"The read reported {logprobs.Count} positions; the readout needs position {position}.",
                    nameof(logprobs));
            DiffusionPositionLogprobs reported = logprobs[position];
            var scores = new float[allowed.Length];
            for (int i = 0; i < allowed.Length; i++)
            {
                int at = Array.IndexOf(reported.TokenIds, allowed[i]);
                if (at < 0)
                    throw new ArgumentException(
                        $"The read did not report token {allowed[i]} at position {position}.",
                        nameof(logprobs));
                if (!float.IsFinite(reported.Logprobs[at]))
                    throw new ArgumentException(
                        $"Non-finite score for token {allowed[i]} at position {position}.", nameof(logprobs));
                scores[i] = reported.Logprobs[at];
            }
            return scores;
        }

        /// <summary>Decode the selected ids and check the result really is a member of the allowed
        /// language. No JSON repair and no second model pass: if this fails the canvas was wrong.</summary>
        private string Decode(List<int> ids, Dictionary<string, StructuredFieldAnswer> fields)
        {
            string text = _tokenizer.Decode(ids);
            if (JsonNode.Parse(text) is not JsonObject parsed)
                throw new InvalidOperationException($"The selected canvas is not a JSON object: {text}");
            if (!parsed.Select(x => x.Key).SequenceEqual(Questions.Keys))
                throw new InvalidOperationException($"The selected canvas has the wrong keys: {text}");
            foreach ((string key, StructuredFieldAnswer answer) in fields)
            {
                if (JsonValue.Serialize(JsonValue.Parse(parsed[key])) != JsonValue.Serialize(answer.Value))
                    throw new InvalidOperationException(
                        $"The selected canvas disagrees with the readout on '{key}': {text}");
            }
            return text;
        }
    }
}
