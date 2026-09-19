// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// The host half of a DiffusionGemma structured read: what a read is allowed to ask for, and the
// per-position distribution it reads back. Both are model-free, so they run in ordinary CI rather than
// behind the 16 GB checkpoint gate that DiffusionGemmaTests sits behind.
using System;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class DiffusionStructuredReadTests
{
    private const int ServedCanvas = 64;
    private const int Vocab = 512;

    private static DiffusionReadOptions Read(Action<DiffusionReadOptions> tweak = null)
    {
        var o = new DiffusionReadOptions { ReadOnly = true, MaxSteps = 1 };
        tweak?.Invoke(o);
        return o;
    }

    // ---- What a read may ask for -------------------------------------------

    [Fact]
    public void AWellFormedRead_Validates()
    {
        var o = Read(x =>
        {
            x.CanvasWidth = 8;
            x.SeedCanvas = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            x.TopLogprobs = 4;
        });
        o.Validate(ServedCanvas, Vocab);
        Assert.Equal(8, o.EffectiveWidth(ServedCanvas));
    }

    [Fact]
    public void NoWidth_MeansTheServedCanvas()
    {
        var o = Read();
        Assert.Equal(ServedCanvas, o.EffectiveWidth(ServedCanvas));

        // …and a seed canvas is then measured against the served canvas, not against nothing.
        o.SeedCanvas = new int[ServedCanvas - 1];
        Assert.Throws<ArgumentException>(() => o.Validate(ServedCanvas, Vocab));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(ServedCanvas + 1)]
    public void ACanvasWiderThanServed_OrEmpty_IsRefused(int width)
    {
        var o = Read(x => x.CanvasWidth = width);
        Assert.Throws<ArgumentException>(() => o.Validate(ServedCanvas, Vocab));
    }

    [Fact]
    public void ASeedCanvasThatDoesNotMatchItsWidth_IsRefused()
    {
        // Slot positions are the whole contract of a read: a canvas of the wrong length would silently
        // read back different slots than the caller wrote.
        var o = Read(x => { x.CanvasWidth = 4; x.SeedCanvas = new[] { 1, 2, 3 }; });
        Assert.Throws<ArgumentException>(() => o.Validate(ServedCanvas, Vocab));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(Vocab)]
    public void ASeedIdOutsideTheVocabulary_IsRefused(int id)
    {
        // On a GPU this is an out-of-bounds embedding lookup, i.e. a device-side assert.
        var o = Read(x => { x.CanvasWidth = 2; x.SeedCanvas = new[] { 0, id }; });
        Assert.Throws<ArgumentException>(() => o.Validate(ServedCanvas, Vocab));
    }

    [Fact]
    public void AStepCapBelowOne_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => Read(x => x.MaxSteps = 0).Validate(ServedCanvas, Vocab));
    }

    [Fact]
    public void LogprobsWithoutARead_IsRefused_AndOutOfRangeToo()
    {
        Assert.Throws<ArgumentException>(() =>
            Read(x => { x.ReadOnly = false; x.TopLogprobs = 2; }).Validate(ServedCanvas, Vocab));
        Assert.Throws<ArgumentException>(() =>
            Read(x => x.TopLogprobs = Vocab + 1).Validate(ServedCanvas, Vocab));
    }

    // ---- What a read does to the sampler -----------------------------------

    [Fact]
    public void ApplyTo_PinsTheRequestToOneCanvas()
    {
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, MaxBlocks = 8 };
        Read(x =>
        {
            x.CanvasWidth = 16;
            x.MaxSteps = 2;
            x.SeedCanvas = new int[16];
            x.TopLogprobs = 5;
        }).ApplyTo(p, ServedCanvas);

        Assert.Equal(16, p.CanvasWidth);
        Assert.Equal(2, p.MaxDenoisingSteps);
        Assert.Equal(16, p.SeedCanvas.Length);
        Assert.True(p.ReadOnly);
        Assert.Equal(5, p.TopLogprobs);
        // One canvas is the whole output of a read, so no second block is generated.
        Assert.Equal(1, p.MaxBlocks);
    }

    [Fact]
    public void ApplyTo_LeavesAnOrdinaryGenerationAlone()
    {
        var p = new DiffusionEbParams { MaxDenoisingSteps = 48, MaxBlocks = 4 };
        new DiffusionReadOptions().ApplyTo(p, ServedCanvas);

        Assert.Equal(ServedCanvas, p.CanvasWidth);
        Assert.Equal(48, p.MaxDenoisingSteps);
        Assert.Null(p.SeedCanvas);
        Assert.False(p.ReadOnly);
        Assert.Equal(4, p.MaxBlocks);
    }

    // ---- The distribution a read reads back --------------------------------

    [Fact]
    public void TopK_RanksByProbability_AndReportsTemperatureOneLogprobs()
    {
        // softmax([0, ln2, ln3]) = [1/6, 2/6, 3/6]
        var logits = new[] { 0f, MathF.Log(2f), MathF.Log(3f) };
        var top = DiffusionLogprobs.TopK(logits, 2);

        Assert.Equal(new[] { 2, 1 }, top.TokenIds);
        Assert.Equal(MathF.Log(3f / 6f), top.Logprobs[0], 4);
        Assert.Equal(MathF.Log(2f / 6f), top.Logprobs[1], 4);
    }

    [Fact]
    public void PinsNeedSomethingToPinTo_AndMustCoverTheCanvas()
    {
        // A pin holds a position at its seed value, so a pin mask without a seed canvas pins to nothing.
        Assert.Throws<ArgumentException>(() =>
            Read(x => x.PinnedPositions = new bool[ServedCanvas]).Validate(ServedCanvas, Vocab));
        Assert.Throws<ArgumentException>(() =>
            Read(x => { x.SeedCanvas = new int[ServedCanvas]; x.PinnedPositions = new bool[4]; })
                .Validate(ServedCanvas, Vocab));

        Read(x => { x.SeedCanvas = new int[ServedCanvas]; x.PinnedPositions = new bool[ServedCanvas]; })
            .Validate(ServedCanvas, Vocab);
    }

    [Fact]
    public void RequestedLogprobIdsAreCheckedLikeAnySeed_AndReachTheSampler()
    {
        var o = Read(x =>
        {
            x.CanvasWidth = 2;
            x.LogprobTokenIds = new[] { new[] { 1, 2 }, null };
        });
        o.Validate(ServedCanvas, Vocab);

        Assert.Throws<ArgumentException>(() => Read(x =>
        {
            x.CanvasWidth = 2;
            x.LogprobTokenIds = new[] { new[] { Vocab } , null };
        }).Validate(ServedCanvas, Vocab));
        // One row per canvas position, so a row index IS a position.
        Assert.Throws<ArgumentException>(() => Read(x =>
        {
            x.CanvasWidth = 2;
            x.LogprobTokenIds = new[] { new[] { 1 } };
        }).Validate(ServedCanvas, Vocab));
        Assert.Throws<ArgumentException>(() => Read(x =>
        {
            x.ReadOnly = false;
            x.LogprobTokenIds = new int[ServedCanvas][];
        }).Validate(ServedCanvas, Vocab));

        var p = new DiffusionEbParams();
        o.ApplyTo(p, ServedCanvas);
        Assert.Same(o.LogprobTokenIds, p.LogprobTokenIds);
    }

    [Fact]
    public void ScoresFor_ReportsTheAskedTokensInOrder_EvenFarOutsideTheTopK()
    {
        // A constrained readout needs the score of an allowed token wherever it ranks, so this asks by id.
        var logits = new float[Vocab];
        logits[7] = 10f;
        var scores = DiffusionLogprobs.ScoresFor(logits, new[] { 300, 7, 42 });

        Assert.Equal(new[] { 300, 7, 42 }, scores.TokenIds);
        Assert.True(scores.Logprobs[1] > scores.Logprobs[0]);
        Assert.Equal(scores.Logprobs[0], scores.Logprobs[2], 5);
        // Same scale as the top-K path: both are log-softmax over the whole vocabulary.
        Assert.Equal(DiffusionLogprobs.TopK(logits, 1).Logprobs[0], scores.Logprobs[1], 5);
    }

    [Fact]
    public void PerPosition_AsksWhereTold_AndRanksWhereNot()
    {
        const int vocab = 4;
        var logits = new float[3 * vocab];
        logits[0 * vocab + 3] = 5f;
        logits[1 * vocab + 1] = 5f;
        logits[2 * vocab + 2] = 5f;

        var perPos = DiffusionLogprobs.PerPosition(
            logits, width: 3, vocab: vocab, k: 2,
            requested: new[] { new[] { 0, 3 }, null, Array.Empty<int>() });

        Assert.Equal(new[] { 0, 3 }, perPos[0].TokenIds);   // asked for: exactly these, in order
        Assert.Equal(1, perPos[1].TokenIds[0]);             // not asked: the position's own top-k
        Assert.Equal(2, perPos[1].TokenIds.Length);
        Assert.Equal(2, perPos[2].TokenIds[0]);             // an empty row is not a request

        // With no top-k either, a position nobody asked about reports nothing.
        var none = DiffusionLogprobs.PerPosition(
            logits, 3, vocab, k: 0, requested: new[] { new[] { 1 }, null, null });
        Assert.Single(none[0].TokenIds);
        Assert.Empty(none[1].TokenIds);
    }

    [Fact]
    public void TopK_SumsToOne_OverTheWholeVocabulary()
    {
        var rng = new Random(7);
        var logits = new float[Vocab];
        for (int i = 0; i < Vocab; i++) logits[i] = (float)(rng.NextDouble() * 8 - 4);

        var top = DiffusionLogprobs.TopK(logits, Vocab);
        double mass = 0;
        foreach (float lp in top.Logprobs) mass += Math.Exp(lp);
        Assert.Equal(1.0, mass, 3);
    }

    [Fact]
    public void TopK_LeadsWithTheArgmax_TieBrokenTowardsTheLowerId()
    {
        // The emitted canvas is the argmax canvas, so the reported distribution has to agree with it.
        var top = DiffusionLogprobs.TopK(new[] { 1f, 1f, 0f }, 3);
        Assert.Equal(new[] { 0, 1, 2 }, top.TokenIds);
        Assert.True(top.Logprobs[0] >= top.Logprobs[1] && top.Logprobs[1] >= top.Logprobs[2]);
    }

    [Fact]
    public void TopK_IsClampedToTheVocabulary()
    {
        var top = DiffusionLogprobs.TopK(new[] { 1f, 2f }, 99);
        Assert.Equal(2, top.TokenIds.Length);
        Assert.Equal(new[] { 1, 0 }, top.TokenIds);
    }

    [Fact]
    public void TopKPerPosition_ReadsEachCanvasSlotOnItsOwn_AndStopsAtTheWidth()
    {
        const int vocab = 4;
        // Three canvas positions, each peaking on a different token; the read only owns the first two.
        var logits = new float[3 * vocab];
        logits[0 * vocab + 3] = 5f;
        logits[1 * vocab + 1] = 5f;
        logits[2 * vocab + 0] = 5f;

        var perPos = DiffusionLogprobs.TopKPerPosition(logits, width: 2, vocab: vocab, k: 1);
        Assert.Equal(2, perPos.Length);
        Assert.Equal(3, perPos[0].TokenIds[0]);
        Assert.Equal(1, perPos[1].TokenIds[0]);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => DiffusionLogprobs.TopKPerPosition(logits, width: 4, vocab: vocab, k: 1));
    }
}
