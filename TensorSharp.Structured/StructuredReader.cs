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
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured.Decisions;

namespace TensorSharp.Structured
{
    /// <summary>One canvas to denoise: the prompt that conditions it and the read that shapes it.</summary>
    public sealed class StructuredRead
    {
        public required string Prompt { get; init; }
        public required DiffusionReadOptions Options { get; init; }
        public int Seed { get; init; }
    }

    /// <summary>
    /// What a structured prediction needs from a model: the canvas geometry to compile against, a
    /// tokenizer to compile with, and a way to denoise a batch of seeded canvases and report the scores
    /// at their free positions.
    ///
    /// The interface takes a batch because batching is where the throughput is - a canvas is a single
    /// forward pass, so several of them cost barely more than one.
    /// </summary>
    public interface IStructuredReader
    {
        ITokenizer Tokenizer { get; }

        /// <summary>Canvas positions the model denoises per forward pass.</summary>
        int CanvasLength { get; }

        int VocabSize { get; }

        /// <summary>The token the canvas is padded with past its JSON.</summary>
        int EosTokenId { get; }

        /// <summary>Denoise every canvas in the batch and report each one's per-position scores. Results
        /// are aligned with <paramref name="reads"/>.</summary>
        Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
            IReadOnlyList<StructuredRead> reads, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// <see cref="IStructuredReader"/> over a loaded DiffusionGemma model, batching a whole set of
    /// canvases into one denoising block the way the server's scheduler batches chat turns.
    ///
    /// Each read holds the model's compute lock for its block, so concurrent calls serialize; hand this
    /// the whole batch rather than calling it once per request.
    /// </summary>
    public sealed class DiffusionGemmaReader : IStructuredReader, IDecisionReader
    {
        private static readonly IPromptRenderer Renderer = new GgufPromptRenderer();

        private readonly DiffusionGemmaModel _model;
        private readonly DiffusionGemmaSampler _sampler;
        private readonly DiffusionEbParams _defaults;

        /// <param name="model">A loaded DiffusionGemma model.</param>
        /// <param name="defaults">Sampler hyper-parameters the reads start from; the read's own fields
        /// (width, seed canvas, step cap) are applied on top. Null takes the sampler defaults.</param>
        public DiffusionGemmaReader(DiffusionGemmaModel model, DiffusionEbParams? defaults = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _sampler = new DiffusionGemmaSampler(model);
            _defaults = defaults ?? new DiffusionEbParams();
        }

        public ITokenizer Tokenizer => _model.Tokenizer;
        public int CanvasLength => _model.CanvasLength;
        public int VocabSize => _model.VocabSize;
        public int EosTokenId => _model.Tokenizer.EosTokenIds is { Length: > 0 } eos ? eos[0] : 0;

        /// <summary>The checkpoint's declared context, else djev's default of 32768.</summary>
        public int MaxContextLength => _model.Config.DeclaredContextLength > 0 ? _model.Config.DeclaredContextLength : 32768;

        public string ModelName => "diffusiongemma";

        /// <inheritdoc/>
        public int[] EncodeChat(string system, string user) =>
            _model.Tokenizer.Encode(
                DecisionPrompt.Render(_model.Config.ChatTemplate, _model.Config.Architecture, system, user),
                addSpecial: true).ToArray();

        /// <inheritdoc/>
        public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
            IReadOnlyList<DecisionRead> reads, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(reads);
            if (reads.Count == 0)
                return Task.FromResult<IReadOnlyList<DiffusionReadResult>>(Array.Empty<DiffusionReadResult>());
            var batch = new List<(int[] Prompt, DiffusionReadOptions Options, int Seed)>(reads.Count);
            foreach (DecisionRead read in reads) batch.Add((read.PromptTokens, read.Options, read.SamplerSeed));
            return Task.Run<IReadOnlyList<DiffusionReadResult>>(() => Read(batch, cancellationToken), cancellationToken);
        }

        public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
            IReadOnlyList<StructuredRead> reads, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(reads);
            if (reads.Count == 0)
                return Task.FromResult<IReadOnlyList<DiffusionReadResult>>(Array.Empty<DiffusionReadResult>());

            // The sampler is synchronous and owns the GPU for the block; run it off the caller's thread so
            // an async pipeline is not blocked by it.
            return Task.Run<IReadOnlyList<DiffusionReadResult>>(() =>
            {
                var batch = new List<(int[] Prompt, DiffusionReadOptions Options, int Seed)>(reads.Count);
                foreach (StructuredRead read in reads) batch.Add((Tokenize(read.Prompt), read.Options, read.Seed));
                return Read(batch, cancellationToken);
            }, cancellationToken);
        }

        private IReadOnlyList<DiffusionReadResult> Read(
            IReadOnlyList<(int[] Prompt, DiffusionReadOptions Options, int Seed)> reads,
            CancellationToken cancellationToken)
        {
            var runs = new List<DiffusionSeqRun>(reads.Count);
            var states = new List<DiffusionSeqState>(reads.Count);
            try
            {
                lock (_model.GpuComputeLock)
                {
                    foreach ((int[] prompt, DiffusionReadOptions options, int seed) in reads)
                    {
                        options.Validate(CanvasLength, VocabSize);
                        var parameters = Clone(_defaults);
                        parameters.Seed = seed;
                        options.ApplyTo(parameters, CanvasLength);

                        DiffusionSeqState state = _model.CreateSeqState();
                        states.Add(state);
                        runs.Add(new DiffusionSeqRun(
                            prompt, parameters, state, cancellationToken, onPreview: null!));
                    }

                    // A read ends on the canvas it emits, so one block answers the whole batch.
                    _sampler.RunBlockBatched(runs, cancellationToken);
                }

                var results = new DiffusionReadResult[runs.Count];
                for (int i = 0; i < runs.Count; i++)
                {
                    results[i] = runs[i].ReadResult
                        ?? throw new InvalidOperationException(
                            "The sampler produced no read result; the request was not read-only.");
                }
                return results;
            }
            finally
            {
                lock (_model.GpuComputeLock)
                {
                    foreach (DiffusionSeqState state in states)
                    {
                        try { _model.DisposeSeqState(state); }
                        catch (Exception) { /* a failed disposal must not mask the read's own error */ }
                    }
                }
            }
        }

        private int[] Tokenize(string prompt)
        {
            var messages = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
            string rendered = Renderer.Render(
                _model.Config.ChatTemplate, messages,
                addGenerationPrompt: true, architecture: _model.Config.Architecture);
            return _model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
        }

        private static DiffusionEbParams Clone(DiffusionEbParams p) => new()
        {
            MaxDenoisingSteps = p.MaxDenoisingSteps,
            TMin = p.TMin,
            TMax = p.TMax,
            EntropyBound = p.EntropyBound,
            StabilityThreshold = p.StabilityThreshold,
            ConfidenceThreshold = p.ConfidenceThreshold,
            Seed = p.Seed,
            MaxBlocks = p.MaxBlocks,
        };
    }
}
