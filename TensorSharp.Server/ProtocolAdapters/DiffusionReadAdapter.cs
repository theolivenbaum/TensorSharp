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
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>
/// <c>POST /v1/diffusion/read</c> — a structured read of a DiffusionGemma canvas.
///
/// A discrete diffusion model denoises a whole canvas per forward pass, so seeding the canvas with the
/// answer's fixed text and leaving only the answer slots as noise turns one denoise step into a
/// distribution over each slot. That is a different shape from chat completions — there is no
/// continuation, no streaming and no finish reason, and the payload is per-position logprobs — so it gets
/// its own route rather than more flags on <c>/v1/chat/completions</c>.
///
/// The request fields are named after the <c>extra_args</c> that vLLM accepts for the same model, so a
/// schema layer written against either engine drives the other unchanged.
/// </summary>
public sealed class DiffusionReadAdapter
{
    private readonly ModelService _svc;
    private readonly InferenceQueue _queue;
    private readonly ILogger _logger;

    public DiffusionReadAdapter(ModelService svc, InferenceQueue queue, ILoggerFactory loggerFactory)
    {
        _svc = svc ?? throw new ArgumentNullException(nameof(svc));
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
        _logger = (loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory)))
            .CreateLogger("TensorSharp.DiffusionRead");
    }

    public async Task<IResult> ReadAsync(HttpContext ctx)
    {
        if (!_svc.IsLoaded)
            return Error(StatusCodes.Status503ServiceUnavailable, "No model is loaded.");
        if (!_svc.IsDiffusionModel)
            return Error(StatusCodes.Status400BadRequest,
                "Structured reads need a block-diffusion model; the loaded model generates autoregressively.");

        var model = (DiffusionGemmaModel)_svc.Model;

        JsonElement body;
        try
        {
            body = await JsonSerializer.DeserializeAsync<JsonElement>(
                ctx.Request.Body, cancellationToken: ctx.RequestAborted).ConfigureAwait(false);
        }
        catch (JsonException ex)
        {
            return Error(StatusCodes.Status400BadRequest, $"Malformed JSON body: {ex.Message}");
        }

        List<ChatMessage> messages;
        DiffusionReadOptions options;
        int? seed;
        try
        {
            messages = ParseMessages(body);
            options = ParseOptions(body, model);
            seed = body.TryGetProperty("seed", out var s) && s.ValueKind == JsonValueKind.Number
                ? s.GetInt32() : null;
        }
        catch (ArgumentException ex)
        {
            return Error(StatusCodes.Status400BadRequest, ex.Message);
        }

        using var ticket = _queue.Enqueue(ctx.RequestAborted);
        DiffusionReadResult result;
        try
        {
            result = await _svc.DiffusionReadAsync(null!, messages, options, seed, ctx.RequestAborted)
                .ConfigureAwait(false);
        }
        catch (ArgumentException ex)
        {
            // The sampler and the options validator speak the same refusals; both are the caller's fault.
            return Error(StatusCodes.Status400BadRequest, ex.Message);
        }
        catch (OperationCanceledException)
        {
            return Results.StatusCode(StatusCodes.Status499ClientClosedRequest);
        }

        return Results.Json(Serialize(result, model, _svc.LoadedModelName));
    }

    // ---- Request -----------------------------------------------------------

    private static List<ChatMessage> ParseMessages(JsonElement body)
    {
        if (!body.TryGetProperty("messages", out var arr) || arr.ValueKind != JsonValueKind.Array
            || arr.GetArrayLength() == 0)
            throw new ArgumentException("messages must be a non-empty array.");

        var messages = new List<ChatMessage>(arr.GetArrayLength());
        foreach (var m in arr.EnumerateArray())
        {
            string? role = m.TryGetProperty("role", out var r) ? r.GetString() : null;
            string? content = m.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString() : null;
            if (string.IsNullOrEmpty(role) || content == null)
                throw new ArgumentException("every message needs a role and a string content.");
            messages.Add(new ChatMessage { Role = role, Content = content });
        }
        return messages;
    }

    private static DiffusionReadOptions ParseOptions(JsonElement body, DiffusionGemmaModel model)
    {
        var options = new DiffusionReadOptions { ReadOnly = true };

        if (body.TryGetProperty("diffusion_canvas_length", out var w) && w.ValueKind == JsonValueKind.Number)
            options.CanvasWidth = w.GetInt32();
        if (body.TryGetProperty("diffusion_max_steps", out var st) && st.ValueKind == JsonValueKind.Number)
            options.MaxSteps = st.GetInt32();
        if (body.TryGetProperty("top_logprobs", out var k) && k.ValueKind == JsonValueKind.Number)
            options.TopLogprobs = k.GetInt32();

        bool haveIds = body.TryGetProperty("diffusion_seed_canvas", out var seedIds)
            && seedIds.ValueKind == JsonValueKind.Array;
        bool haveText = body.TryGetProperty("diffusion_seed_canvas_text", out var seedText)
            && seedText.ValueKind == JsonValueKind.String;
        if (haveIds && haveText)
            throw new ArgumentException(
                "give the seed canvas as diffusion_seed_canvas (ids) or diffusion_seed_canvas_text, not both.");

        if (haveIds)
        {
            var ids = new int[seedIds.GetArrayLength()];
            int i = 0;
            foreach (var id in seedIds.EnumerateArray())
            {
                if (id.ValueKind != JsonValueKind.Number)
                    throw new ArgumentException("diffusion_seed_canvas must be a list of token ids.");
                ids[i++] = id.GetInt32();
            }
            options.SeedCanvas = ids;
        }
        else if (haveText)
        {
            // The width the read declares is the contract; the tokenizer has to land on it exactly, or
            // the caller's slot positions would not be the ones read back.
            options.SeedCanvas = model.Tokenizer.Encode(seedText.GetString() ?? string.Empty, addSpecial: false).ToArray();
        }

        // Validate here so a malformed read is a 400 rather than a failure deep in the sampler.
        options.Validate(model.CanvasLength, model.VocabSize);
        return options;
    }

    // ---- Response ----------------------------------------------------------

    private static object Serialize(DiffusionReadResult result, DiffusionGemmaModel model, string modelName)
    {
        var positions = new List<object>(result.Logprobs.Count);
        for (int pos = 0; pos < result.Logprobs.Count; pos++)
        {
            var lp = result.Logprobs[pos];
            var top = new List<object>(lp.TokenIds.Length);
            for (int i = 0; i < lp.TokenIds.Length; i++)
            {
                if (lp.TokenIds[i] < 0) continue;
                top.Add(new
                {
                    token = model.Tokenizer.Decode(new List<int> { lp.TokenIds[i] }),
                    token_id = lp.TokenIds[i],
                    logprob = lp.Logprobs[i],
                });
            }
            positions.Add(new { position = pos, top_logprobs = top });
        }

        return new
        {
            @object = "diffusion.read",
            model = modelName,
            canvas = new
            {
                tokens = result.Canvas,
                text = model.Tokenizer.Decode(new List<int>(result.Canvas)),
            },
            steps = result.StepsRun,
            converged = result.Converged,
            logprobs = positions,
        };
    }

    private static IResult Error(int status, string message) =>
        Results.Json(new { error = new { message, type = "invalid_request_error" } }, statusCode: status);
}
