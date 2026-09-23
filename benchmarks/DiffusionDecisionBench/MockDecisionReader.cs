// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured.Decisions;

namespace DiffusionDecisionBench;

/// <summary>
/// A stand-in for the model, so the harness can be exercised end to end without a checkpoint: one token per
/// word, a chat prompt that is just the two turns concatenated, and label scores hashed from the
/// prompt, the slot and the label. It proves compilation, batching, validation, scoring and the receipt -
/// and nothing whatsoever about DiffusionGemma.
/// </summary>
internal sealed class MockDecisionReader : IDecisionReader
{
    /// <summary>Words, digits and single symbols, each one token, ids hashed into the vocabulary - enough
    /// that <c>no</c>/<c>yes</c>, <c>A</c>..<c>Z</c> and <c>0</c>..<c>9</c> are single tokens, as they are for
    /// Gemma.</summary>
    private sealed class WordTokenizer : ITokenizer
    {
        private static readonly System.Text.RegularExpressions.Regex Pieces =
            new(@"<\|channel>|<channel\|>|[A-Za-z]+|\d|\s|.", System.Text.RegularExpressions.RegexOptions.Singleline);
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 2;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => 262144;
        public List<int> Encode(string text, bool addSpecial = true)
        {
            var ids = new List<int>();
            if (addSpecial) ids.Add(BosTokenId);
            foreach (System.Text.RegularExpressions.Match m in Pieces.Matches(text)) ids.Add(Id(m.Value));
            return ids;
        }
        private static int Id(string piece)
        {
            uint h = 2166136261;
            foreach (char c in piece) h = (h ^ c) * 16777619;
            return 300 + (int)(h % (262144 - 300));
        }
        public string Decode(List<int> ids) => string.Join(" ", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => Id(tokenStr);
    }

    public ITokenizer Tokenizer { get; } = new WordTokenizer();
    public int CanvasLength => 256;
    public int VocabSize => 262144;
    public int MaxContextLength => 32768;
    public string ModelName => "mock";

    public int[] EncodeChat(string system, string user) =>
        Tokenizer.Encode($"<system>{system}</system><user>{user}</user><model>", addSpecial: true).ToArray();

    public Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(IReadOnlyList<DecisionRead> reads, CancellationToken cancellationToken = default)
    {
        var results = reads.Select(read =>
        {
            ulong promptHash = 1469598103934665603UL;
            foreach (int t in read.PromptTokens) promptHash = (promptHash ^ (uint)t) * 1099511628211UL;
            int width = read.Options.CanvasWidth ?? CanvasLength;
            var rows = new DiffusionPositionLogprobs[width];
            for (int pos = 0; pos < width; pos++)
            {
                int[] ids = read.Options.LogprobTokenIds?[pos] ?? Array.Empty<int>();
                var scores = ids.Select(id => Score(promptHash, pos, id)).ToArray();
                // Leave 10% of the mass outside the allowed labels, as a real read would.
                double max = scores.DefaultIfEmpty(0).Max();
                double logZ = max + Math.Log(scores.Sum(s => Math.Exp(s - max))) - Math.Log(0.9);
                rows[pos] = new DiffusionPositionLogprobs(ids, scores.Select(s => (float)(s - logZ)).ToArray());
            }
            return new DiffusionReadResult(read.Options.SeedCanvas ?? new int[width], rows, stepsRun: 1, converged: false);
        }).ToList();
        return Task.FromResult<IReadOnlyList<DiffusionReadResult>>(results);
    }

    private static double Score(ulong promptHash, int position, int id)
    {
        ulong h = (promptHash ^ ((ulong)position << 32) ^ (uint)id) * 0x9E3779B97F4A7C15UL;
        h ^= h >> 29;
        return (h % 10_000) / 10_000.0 * 4.0;
    }
}
