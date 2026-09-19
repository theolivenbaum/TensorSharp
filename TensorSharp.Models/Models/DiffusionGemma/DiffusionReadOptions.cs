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

namespace TensorSharp.Models
{
    /// <summary>
    /// A structured read of a DiffusionGemma canvas: the caller writes the answer's fixed text into the
    /// canvas, leaves the slots it wants read as noise, and denoises for a fixed number of steps. What
    /// comes back is the distribution the model puts over every slot rather than a sampled continuation.
    ///
    /// The three fields mirror the <c>diffusion_seed_canvas</c> / <c>diffusion_max_steps</c> /
    /// <c>diffusion_read_only</c> request fields that vLLM exposes for the same model, so a schema layer
    /// written against one engine drives the other unchanged.
    /// </summary>
    public sealed class DiffusionReadOptions
    {
        /// <summary>Token ids that replace the random initial canvas, exactly <see cref="CanvasWidth"/>
        /// long (the served canvas when no width is given). Null leaves the canvas random.</summary>
        public int[] SeedCanvas;

        /// <summary>The leading canvas positions this request owns, at most the served canvas length.
        /// Null (or 0) means the whole served canvas.</summary>
        public int? CanvasWidth;

        /// <summary>Denoising steps before the canvas is emitted. Null keeps the sampler's default.</summary>
        public int? MaxSteps;

        /// <summary>Emit the argmax canvas as soon as the step cap is reached and end the request there:
        /// one canvas is the whole output, no end-of-turn trimming and no further block.</summary>
        public bool ReadOnly;

        /// <summary>How many top tokens to report per canvas position, at temperature 1. 0 reports none.
        /// Only read-only requests report logprobs.</summary>
        public int TopLogprobs;

        /// <summary>The canvas this request denoises: its own width, else the served one.</summary>
        public int EffectiveWidth(int servedCanvasLength) =>
            CanvasWidth is > 0 ? Math.Min(CanvasWidth.Value, servedCanvasLength) : servedCanvasLength;

        /// <summary>
        /// Reject a malformed read before it reaches the sampler. A seed id outside the vocabulary is an
        /// out-of-bounds embedding lookup on the device, and a canvas that does not match its width would
        /// silently read the wrong slots.
        /// </summary>
        /// <param name="servedCanvasLength">The loaded model's canvas length.</param>
        /// <param name="vocabSize">The loaded model's vocabulary size.</param>
        /// <exception cref="ArgumentException">The options are not a valid read of this model.</exception>
        public void Validate(int servedCanvasLength, int vocabSize)
        {
            if (CanvasWidth is { } width && (width < 1 || width > servedCanvasLength))
                throw new ArgumentException(
                    "diffusion_canvas_length must be a positive integer no larger than the served canvas " +
                    $"({servedCanvasLength}).", nameof(CanvasWidth));

            int expectedLen = EffectiveWidth(servedCanvasLength);

            if (SeedCanvas != null)
            {
                foreach (int t in SeedCanvas)
                {
                    if (t < 0 || t >= vocabSize)
                        throw new ArgumentException(
                            $"diffusion_seed_canvas ids must be in [0, {vocabSize}).", nameof(SeedCanvas));
                }
                if (SeedCanvas.Length != expectedLen)
                    throw new ArgumentException(
                        $"diffusion_seed_canvas must hold exactly {expectedLen} ids, got {SeedCanvas.Length}.",
                        nameof(SeedCanvas));
            }

            if (MaxSteps is { } steps && steps < 1)
                throw new ArgumentException("diffusion_max_steps must be a positive integer.", nameof(MaxSteps));

            if (TopLogprobs < 0 || TopLogprobs > vocabSize)
                throw new ArgumentException(
                    $"diffusion_top_logprobs must be in [0, {vocabSize}].", nameof(TopLogprobs));

            if (TopLogprobs > 0 && !ReadOnly)
                throw new ArgumentException(
                    "diffusion_top_logprobs needs diffusion_read_only: only a read emits per-position " +
                    "distributions.", nameof(TopLogprobs));
        }

        /// <summary>
        /// Fold these options into the sampler parameters of a request. Validate first — this applies the
        /// values as given. A read-only request is capped to one block: the canvas it emits is its whole
        /// output.
        /// </summary>
        public void ApplyTo(DiffusionEbParams p, int servedCanvasLength)
        {
            if (p == null) throw new ArgumentNullException(nameof(p));

            p.CanvasWidth = EffectiveWidth(servedCanvasLength);
            p.SeedCanvas = SeedCanvas;
            if (MaxSteps is { } steps) p.MaxDenoisingSteps = Math.Max(1, steps);
            p.ReadOnly = ReadOnly;
            p.TopLogprobs = TopLogprobs;
            if (ReadOnly) p.MaxBlocks = 1;
        }
    }

    /// <summary>The model's distribution over one canvas position, at temperature 1, most likely first.</summary>
    public sealed class DiffusionPositionLogprobs
    {
        /// <summary>Token ids, ordered by descending probability. The first is the emitted argmax.</summary>
        public int[] TokenIds { get; }

        /// <summary>Natural log-probabilities aligned with <see cref="TokenIds"/>.</summary>
        public float[] Logprobs { get; }

        public DiffusionPositionLogprobs(int[] tokenIds, float[] logprobs)
        {
            TokenIds = tokenIds ?? throw new ArgumentNullException(nameof(tokenIds));
            Logprobs = logprobs ?? throw new ArgumentNullException(nameof(logprobs));
        }
    }

    /// <summary>The result of a read-only denoise: the emitted canvas and, when asked for, the
    /// temperature-1 distribution at every one of its positions.</summary>
    public sealed class DiffusionReadResult
    {
        /// <summary>The emitted argmax canvas, <c>CanvasWidth</c> tokens, untrimmed.</summary>
        public int[] Canvas { get; }

        /// <summary>One entry per canvas position, or an empty list when no logprobs were asked for.</summary>
        public IReadOnlyList<DiffusionPositionLogprobs> Logprobs { get; }

        /// <summary>Denoising steps actually run.</summary>
        public int StepsRun { get; }

        /// <summary>True when the canvas met the stability/confidence test rather than hitting the cap.</summary>
        public bool Converged { get; }

        public DiffusionReadResult(int[] canvas, IReadOnlyList<DiffusionPositionLogprobs> logprobs,
            int stepsRun, bool converged)
        {
            Canvas = canvas ?? throw new ArgumentNullException(nameof(canvas));
            Logprobs = logprobs ?? Array.Empty<DiffusionPositionLogprobs>();
            StepsRun = stepsRun;
            Converged = converged;
        }
    }
}
