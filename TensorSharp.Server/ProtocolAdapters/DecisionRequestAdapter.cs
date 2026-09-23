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
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Structured.Decisions;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>
/// <see cref="IDecisionReader"/> over the server's loaded model. Reads go through the diffusion scheduler
/// with already-tokenized prompts, so concurrent decision requests share denoising blocks with each other
/// and with chat traffic, and no prompt is truncated behind the caller's back.
/// </summary>
internal sealed class ServiceDecisionReader : IDecisionReader
{
    private readonly ModelService _svc;
    private readonly DiffusionGemmaModel _model;

    public ServiceDecisionReader(ModelService svc, DiffusionGemmaModel model, string modelName)
    {
        _svc = svc;
        _model = model;
        ModelName = modelName;
    }

    public ITokenizer Tokenizer => _model.Tokenizer;
    public int CanvasLength => _model.CanvasLength;
    public int VocabSize => _model.VocabSize;
    public int MaxContextLength => _model.Config.DeclaredContextLength > 0 ? _model.Config.DeclaredContextLength : 32768;
    public string ModelName { get; }

    public int[] EncodeChat(string system, string user) =>
        _model.Tokenizer.Encode(
            DecisionPrompt.Render(_model.Config.ChatTemplate, _model.Config.Architecture, system, user),
            addSpecial: true).ToArray();

    public async Task<IReadOnlyList<DiffusionReadResult>> ReadAsync(
        IReadOnlyList<DecisionRead> reads, CancellationToken cancellationToken = default)
    {
        // Submitted together, so the scheduler admits them into the same block.
        var pending = reads.Select(r => _svc.DiffusionReadTokensAsync(r.PromptTokens, r.Options, r.SamplerSeed, cancellationToken)).ToArray();
        return await Task.WhenAll(pending).ConfigureAwait(false);
    }
}

/// <summary>
/// djev's decision API on TensorSharp: <c>POST /v1/request</c> takes djev's request body (a state, named
/// Noul / Choice / Score questions, options) and answers with djev's response body, computed the way djev
/// computes it - one-step structured reads of exact label probabilities on a compact seeded canvas.
/// Status codes follow djev: 400 malformed JSON or duplicate keys, 413 too large, 415 not JSON, 422 a
/// request that cannot be answered as asked, 502 incomplete evidence from the model, 503 no model.
/// </summary>
public sealed class DecisionRequestAdapter
{
    /// <summary>djev's <c>MAX_BODY_BYTES</c>.</summary>
    public const int MaxBodyBytes = 8 * 1024 * 1024;

    private readonly ModelService _svc;
    private readonly InferenceQueue _queue;
    private readonly ILogger _logger;
    private readonly object _gate = new();
    private (ModelBase Model, DiffusionAgent Agent)? _agent;

    public DecisionRequestAdapter(ModelService svc, InferenceQueue queue, ILoggerFactory loggerFactory)
    {
        _svc = svc ?? throw new ArgumentNullException(nameof(svc));
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
        _logger = (loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory)))
            .CreateLogger("TensorSharp.DecisionRequest");
    }

    /// <summary>One agent per loaded model, so the compiled-schema cache survives across requests and is
    /// dropped with the model it was compiled against.</summary>
    private DiffusionAgent Agent(DiffusionGemmaModel model)
    {
        lock (_gate)
        {
            if (_agent is { } cached && ReferenceEquals(cached.Model, model)) return cached.Agent;
            var agent = new DiffusionAgent(new ServiceDecisionReader(_svc, model, _svc.LoadedModelName));
            _agent = (model, agent);
            return agent;
        }
    }

    public async Task<IResult> RequestAsync(HttpContext ctx)
    {
        if (!_svc.IsLoaded)
            return Error(StatusCodes.Status503ServiceUnavailable, "No model is loaded.");
        if (!_svc.IsDiffusionModel)
            return Error(StatusCodes.Status400BadRequest,
                "Typed decisions need a block-diffusion model; the loaded model generates autoregressively.");
        string contentType = (ctx.Request.ContentType ?? "").Split(';', 2)[0].Trim();
        if (!string.Equals(contentType, "application/json", StringComparison.OrdinalIgnoreCase))
            return Error(StatusCodes.Status415UnsupportedMediaType, "Content-Type must be application/json");
        if (ctx.Request.ContentLength is > MaxBodyBytes)
            return Error(StatusCodes.Status413PayloadTooLarge, "Request body too large");

        JsonNode? body;
        try
        {
            using var buffer = new MemoryStream();
            var chunk = new byte[81920];
            int read;
            while ((read = await ctx.Request.Body.ReadAsync(chunk, ctx.RequestAborted).ConfigureAwait(false)) > 0)
            {
                if (buffer.Length + read > MaxBodyBytes)
                    return Error(StatusCodes.Status413PayloadTooLarge, "Request body too large");
                buffer.Write(chunk, 0, read);
            }
            buffer.Position = 0;
            body = JsonNode.Parse(buffer, documentOptions: new JsonDocumentOptions { AllowDuplicateProperties = false });
        }
        catch (JsonException ex)
        {
            return Error(StatusCodes.Status400BadRequest, $"Malformed JSON body: {ex.Message}");
        }
        catch (OperationCanceledException)
        {
            return Results.StatusCode(StatusCodes.Status499ClientClosedRequest);
        }

        var model = (DiffusionGemmaModel)_svc.Model;
        using var ticket = _queue.Enqueue(ctx.RequestAborted);
        var started = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            DecisionRequest request = DecisionRequest.FromJson(body ?? throw new DecisionSchemaException("the request body is empty"));
            DecisionResult result = await Agent(model).PredictAsync(request, ctx.RequestAborted).ConfigureAwait(false);
            double serverMs = started.Elapsed.TotalMilliseconds;
            ctx.Response.Headers["Server-Timing"] = string.Create(CultureInfo.InvariantCulture,
                $"compile;dur={result.CompileMs:F2}, model;dur={result.ModelMs:F2}, server;dur={serverMs:F2}");
            ctx.Response.Headers["X-Djev-Model-Ms"] = result.ModelMs.ToString("F2", CultureInfo.InvariantCulture);
            ctx.Response.Headers["X-Djev-Server-Ms"] = serverMs.ToString("F2", CultureInfo.InvariantCulture);
            ctx.Response.Headers["Cache-Control"] = "no-store";
            return Results.Content(result.ToJsonString(), "application/json");
        }
        catch (DecisionSchemaException ex)
        {
            return Error(StatusCodes.Status422UnprocessableEntity, ex.Message);
        }
        catch (DecisionBackendException ex)
        {
            _logger.LogWarning("decision read returned invalid evidence: {Message}", ex.Message);
            return Error(StatusCodes.Status502BadGateway, ex.Message);
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException or ArgumentException)
        {
            // A malformed field that parsed as JSON but not as the contract (a string where a number goes).
            return Error(StatusCodes.Status422UnprocessableEntity, ex.Message);
        }
        catch (OperationCanceledException)
        {
            return Results.StatusCode(StatusCodes.Status499ClientClosedRequest);
        }
    }

    /// <summary>djev's <c>GET /config</c>: what the endpoint accepts.</summary>
    public IResult Config() => Results.Json(new
    {
        api_path = "/v1/request",
        model = _svc.IsLoaded ? _svc.LoadedModelName : null,
        limits = new
        {
            state_characters = DecisionContracts.MaxStateCharacters,
            instructions_characters = DecisionContracts.MaxInstructionsCharacters,
            criterion_characters = DecisionContracts.MaxCriterionCharacters,
            questions = DecisionContracts.MaxQuestions,
            choice_options = DecisionContracts.MaxChoiceOptions,
            score_levels = DecisionContracts.MaxScoreLevels,
            body_bytes = MaxBodyBytes,
        },
        features = new { images = false, question_images = false, durable_requests = false },
    });

    private static IResult Error(int status, string message) =>
        Results.Json(new { error = new { message } }, statusCode: status);
}
