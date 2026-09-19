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
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Per-position distributions read off a denoised canvas.
    ///
    /// A read reports at temperature 1, not at the step's position on the denoising schedule: the schedule
    /// exists to make sampling converge, and tempering the reported numbers would rescale the very
    /// probabilities the caller asked for. The argmax is the same either way, so the emitted canvas and the
    /// reported distribution always agree on the most likely token.
    /// </summary>
    public static class DiffusionLogprobs
    {
        /// <summary>
        /// The <paramref name="k"/> most likely tokens of one position's logits and their natural
        /// log-probabilities, most likely first. Ties break towards the lower token id, matching argmax.
        /// </summary>
        public static DiffusionPositionLogprobs TopK(ReadOnlySpan<float> rowLogits, int k)
        {
            int vocab = rowLogits.Length;
            if (vocab == 0) throw new ArgumentException("Empty logits row.", nameof(rowLogits));
            k = Math.Clamp(k, 1, vocab);

            var ids = new int[k];
            var scores = new float[k];
            ids.AsSpan().Fill(-1);
            scores.AsSpan().Fill(float.NegativeInfinity);

            // Insertion into a k-sized descending run; k is small (tens) next to the vocabulary, and the
            // threshold test rejects all but a handful of the rows.
            for (int v = 0; v < vocab; v++)
            {
                float z = rowLogits[v];
                if (!(z > scores[k - 1])) continue;
                int i = k - 1;
                while (i > 0 && z > scores[i - 1]) { scores[i] = scores[i - 1]; ids[i] = ids[i - 1]; i--; }
                scores[i] = z;
                ids[i] = v;
            }

            // log softmax at the kept entries: z - (m + log sum exp(z - m)).
            float m = scores[0];
            float sumExp = 0f;
            for (int v = 0; v < vocab; v++) sumExp += MathF.Exp(rowLogits[v] - m);
            float logZ = m + MathF.Log(sumExp);
            for (int i = 0; i < k; i++)
                scores[i] = ids[i] < 0 ? float.NegativeInfinity : scores[i] - logZ;

            return new DiffusionPositionLogprobs(ids, scores);
        }

        /// <summary>
        /// <see cref="TopK"/> over the first <paramref name="width"/> positions of a
        /// <c>[canvasLength, vocab]</c> row-major logits buffer, one entry per position.
        /// </summary>
        public static DiffusionPositionLogprobs[] TopKPerPosition(float[] logits, int width, int vocab, int k)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (width < 0 || (long)width * vocab > logits.LongLength)
                throw new ArgumentOutOfRangeException(nameof(width),
                    "The logits buffer is shorter than the requested canvas width.");

            var result = new DiffusionPositionLogprobs[width];
            Parallel.For(0, width, pos =>
            {
                result[pos] = TopK(new ReadOnlySpan<float>(logits, pos * vocab, vocab), k);
            });
            return result;
        }
    }
}
