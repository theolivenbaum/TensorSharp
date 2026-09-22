// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>One read: the exact prompt token ids and the seeded canvas to denoise over them.</summary>
    public sealed class DecisionRead
    {
        /// <summary>The rendered, tokenized chat prompt - what djev's <c>token_ids</c> transport sends.</summary>
        public required int[] PromptTokens { get; init; }

        public required DiffusionReadOptions Options { get; init; }

        /// <summary>The sampler's own noise seed. A one-step read over a fully seeded canvas reports
        /// logits that do not depend on it; it only picks what the emitted argmax canvas re-noises.</summary>
        public int SamplerSeed { get; init; }
    }

    /// <summary>
    /// What djev's decision engine needs from a model: a tokenizer to compile labels with, the chat prompt
    /// rendered the way the model's template renders it, and a batched one-step read that reports the
    /// temperature-1 log-probabilities of requested token ids at requested positions.
    ///
    /// This is the contract of the patched vLLM endpoint djev calls, restated in-process.
    /// </summary>
    public interface IDecisionReader
    {
        ITokenizer Tokenizer { get; }

        /// <summary>The widest canvas the model denoises.</summary>
        int CanvasLength { get; }

        int VocabSize { get; }

        /// <summary>The longest prompt plus canvas the model accepts.</summary>
        int MaxContextLength { get; }

        /// <summary>A short name for the model, reported in results.</summary>
        string ModelName { get; }

        /// <summary>
        /// Render <c>[system, user]</c> through the model's chat template with a generation prompt and
        /// thinking disabled, and tokenize it - djev's <c>apply_chat_template(..., enable_thinking=False)</c>.
        /// </summary>
        int[] EncodeChat(string system, string user);

        /// <summary>Denoise every canvas and report its requested scores. Results align with
        /// <paramref name="reads"/>.</summary>
        Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
            IReadOnlyList<DecisionRead> reads, CancellationToken cancellationToken = default);
    }
}
