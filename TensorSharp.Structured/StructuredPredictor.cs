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
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;

namespace TensorSharp.Structured
{
    /// <summary>How a batch of typed decisions is read.</summary>
    public sealed class StructuredPredictOptions
    {
        /// <summary>Denoising steps before the readout. One step already answers; more can change the
        /// answer, at close to the same cost for a single canvas.</summary>
        public int Steps { get; init; } = 1;

        /// <summary>Noise seed for the free slots. Fixing it makes a prediction reproducible.</summary>
        public int Seed { get; init; }

        /// <summary>Canvases denoised together in one forward pass. The whole batch is one pass, so this
        /// is the throughput knob; too large a value runs the device out of memory.</summary>
        public int BatchSize { get; init; } = 16;
    }

    /// <summary>
    /// Typed JSON decisions on a block-diffusion model.
    ///
    /// The model is never asked to write an answer and then parsed; it is handed a canvas that can only
    /// hold answers, and the readout picks among them. So every prediction is a complete, valid member of
    /// the request's allowed language - there is no JSON repair, no retry and no second pass - and one
    /// denoising step answers every question about a document at once.
    ///
    /// Questions that do not fit one canvas are split across several and merged back, which changes what
    /// each question attends to: questions on the same canvas see each other. This does not isolate
    /// questions from one another.
    /// </summary>
    public sealed class StructuredPredictor
    {
        private readonly IStructuredReader _reader;

        public StructuredPredictor(IStructuredReader reader) =>
            _reader = reader ?? throw new ArgumentNullException(nameof(reader));

        /// <summary>Answer one request.</summary>
        public async Task<StructuredPrediction> PredictAsync(
            StructuredRequest request,
            StructuredPredictOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            IReadOnlyList<StructuredPrediction> predictions =
                await PredictAsync(new[] { request }, options, cancellationToken).ConfigureAwait(false);
            return predictions[0];
        }

        /// <summary>
        /// Answer a batch of requests. Requests need not share a schema: each is compiled to its own
        /// canvas (or canvases), and canvases are batched together up to
        /// <see cref="StructuredPredictOptions.BatchSize"/>.
        /// </summary>
        public async Task<IReadOnlyList<StructuredPrediction>> PredictAsync(
            IReadOnlyList<StructuredRequest> requests,
            StructuredPredictOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(requests);
            options ??= new StructuredPredictOptions();
            if (options.Steps < 1)
                throw new ArgumentException("Steps must be positive.", nameof(options));
            if (options.BatchSize < 1)
                throw new ArgumentException("Batch size must be positive.", nameof(options));

            List<CanvasJob> jobs = requests.SelectMany(Plan).ToList();
            var results = new DiffusionReadResult[jobs.Count];
            for (int offset = 0; offset < jobs.Count; offset += options.BatchSize)
            {
                List<CanvasJob> chunk = jobs
                    .Skip(offset).Take(Math.Min(options.BatchSize, jobs.Count - offset)).ToList();
                IReadOnlyList<DiffusionReadResult> read = await _reader.ReadAsync(
                    chunk.Select(job => ToRead(job, options)).ToList(), cancellationToken)
                    .ConfigureAwait(false);
                for (int i = 0; i < chunk.Count; i++) results[offset + i] = read[i];
            }
            return Merge(requests, jobs, results, options.Steps);
        }

        /// <summary>The canvases a batch would denoise, without running anything. Useful for sizing a
        /// benchmark and for seeing where a request had to be split.</summary>
        public IReadOnlyList<int> PlanCanvasCounts(IReadOnlyList<StructuredRequest> requests) =>
            requests.Select(r => Plan(r).Count).ToList();

        // ---- Planning ----------------------------------------------------------

        /// <summary>One canvas: the questions it carries, its compiled layout and its prompt.</summary>
        private sealed record CanvasJob(StructuredRequest Request, JsonCanvasLayout Layout, string Prompt);

        /// <summary>
        /// Pack a request's questions into as few canvases as they fit in, greedily and in key order.
        /// A question is added while the canvas still compiles; the first one that does not starts a new
        /// canvas. A single question too large for an empty canvas is an error - nothing can carry it.
        /// </summary>
        private List<CanvasJob> Plan(StructuredRequest request)
        {
            request.Validate();
            var jobs = new List<CanvasJob>();
            var packed = new Dictionary<string, StructuredQuestion>();
            JsonCanvasLayout? compiled = null;

            foreach ((string key, StructuredQuestion question) in request.Questions)
            {
                var trial = new Dictionary<string, StructuredQuestion>(packed) { [key] = question };
                JsonCanvasLayout layout;
                try
                {
                    layout = Compile(trial);
                }
                catch (ArgumentException) when (packed.Count > 0)
                {
                    jobs.Add(new CanvasJob(request, compiled!, BuildPrompt(request.Document, packed)));
                    packed = new Dictionary<string, StructuredQuestion> { [key] = question };
                    compiled = Compile(packed);
                    continue;
                }
                packed = trial;
                compiled = layout;
            }
            if (packed.Count > 0)
                jobs.Add(new CanvasJob(request, compiled!, BuildPrompt(request.Document, packed)));
            return jobs;
        }

        private JsonCanvasLayout Compile(IReadOnlyDictionary<string, StructuredQuestion> questions) =>
            JsonCanvasLayout.Compile(
                _reader.Tokenizer, questions, _reader.CanvasLength, _reader.EosTokenId);

        private static StructuredRead ToRead(CanvasJob job, StructuredPredictOptions options) => new()
        {
            Prompt = job.Prompt,
            Seed = options.Seed,
            Options = new DiffusionReadOptions
            {
                ReadOnly = true,
                MaxSteps = options.Steps,
                SeedCanvas = job.Layout.SeedCanvas,
                PinnedPositions = job.Layout.PinnedPositions,
                LogprobTokenIds = job.Layout.LogprobTokenIds,
            },
        };

        private static IReadOnlyList<StructuredPrediction> Merge(
            IReadOnlyList<StructuredRequest> requests,
            List<CanvasJob> jobs,
            DiffusionReadResult[] results,
            int steps)
        {
            var byRequest = new Dictionary<StructuredRequest, List<StructuredPrediction>>();
            for (int i = 0; i < jobs.Count; i++)
            {
                CanvasJob job = jobs[i];
                StructuredPrediction part = job.Layout.Select(
                    job.Request.Id, results[i].Logprobs, steps, canvases: 1);
                if (!byRequest.TryGetValue(job.Request, out List<StructuredPrediction>? parts))
                    byRequest[job.Request] = parts = new List<StructuredPrediction>();
                parts.Add(part);
            }

            var merged = new List<StructuredPrediction>(requests.Count);
            foreach (StructuredRequest request in requests)
            {
                List<StructuredPrediction> parts = byRequest[request];
                if (parts.Count == 1) { merged.Add(parts[0]); continue; }

                // A split request answers in several JSON objects; the caller asked one question set, so
                // the fields are recombined into one, in the order the request declared them.
                var fields = new Dictionary<string, StructuredFieldAnswer>();
                foreach (StructuredPrediction part in parts)
                    foreach ((string key, StructuredFieldAnswer answer) in part.Fields)
                        fields[key] = answer;
                var ordered = request.Questions.Keys.ToDictionary(k => k, k => fields[k]);
                merged.Add(new StructuredPrediction
                {
                    Id = request.Id,
                    Json = Recombine(ordered),
                    Fields = ordered,
                    Steps = steps,
                    Canvases = parts.Count,
                });
            }
            return merged;
        }

        private static string Recombine(IReadOnlyDictionary<string, StructuredFieldAnswer> fields)
        {
            var json = new StringBuilder("{\n");
            bool first = true;
            foreach ((string key, StructuredFieldAnswer answer) in fields)
            {
                if (!first) json.Append(",\n");
                first = false;
                json.Append(JsonValue.Serialize(key)).Append(": ").Append(JsonValue.Serialize(answer.Value));
            }
            return json.Append("\n}").ToString();
        }

        // ---- Prompt ------------------------------------------------------------

        /// <summary>
        /// The document, then the questions and their allowed values.
        ///
        /// The delimiters are formatting, so the model can tell the text it is judging from the
        /// instructions about it. They are not a security boundary: a document that writes
        /// <c>&lt;/user_text&gt;</c> is not stopped from doing so.
        /// </summary>
        internal static string BuildPrompt(
            string document, IReadOnlyDictionary<string, StructuredQuestion> questions)
        {
            var instructions = new StringBuilder(
                "Answer all questions below. Return one JSON object with exactly the listed keys and " +
                "allowed values. No explanations.\n");
            foreach ((string key, StructuredQuestion question) in questions)
            {
                instructions.Append(JsonValue.Serialize(key)).Append(": ")
                    .Append(question.Instructions).Append('\n');
                if (question.Criteria is { Count: > 0 } criteria)
                {
                    instructions.Append("Criteria: ")
                        .Append(JsonValue.Serialize(criteria.ToArray())).Append('\n');
                }
                instructions.Append("Allowed values: ")
                    .Append(JsonValue.Serialize(question.Options.ToArray())).Append('\n');
            }
            return "<user_text>\n" + document + "\n</user_text>\n\n<instructions>\n"
                + instructions + "</instructions>";
        }
    }
}
