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
// What djev sends DiffusionGemma for every public JevBench decision, reproduced here.
//
// Fixtures/Djev/gemma_fixtures.json was produced by djev's own engine with the tokenizer djev pins
// (google/diffusiongemma-26B-A4B-it@f7f5b7f5, transformers 5.17.0); gemma_chat_template.jinja is that
// revision's chat template. The prompt TEXT is checked without a checkpoint - the system prompt does not depend
// on the tokenizer, and TensorSharp's Jinja engine renders the template - and the token ids, the answer
// template and the seeded canvas are checked against a real GGUF when TS_TEST_MODEL_DIR names one.
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Structured.Decisions;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class DjevGemmaParityTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string GgufPattern = "diffusion-gemma|gemma-diffusion|diffusiongemma|gemmadiffusion";

    private readonly ITestOutputHelper _output;

    public DjevGemmaParityTests(ITestOutputHelper output) => _output = output;

    private static string Resource(string name)
    {
        using Stream stream = typeof(DjevGemmaParityTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Djev." + name)!;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static JsonArray Tasks() => JsonNode.Parse(Resource("gemma_fixtures.json"))!["tasks"]!.AsArray();

    private static DecisionRequest RequestOf(JsonNode record) => DecisionRequest.FromJson(record["request"]!);

    [Fact]
    public void RendersTheSamePromptTextAsDjev()
    {
        string template = Resource("gemma_chat_template.jinja");
        using var agent = new DiffusionAgent(new DjevDecisionTests.RecordingReader());
        int checkedPrompts = 0;
        foreach (JsonNode? record in Tasks())
        {
            if (record?["prompt_text"] is not { } expected) continue;
            DecisionRequest request = RequestOf(record);
            string system = agent.Compile(request.Questions).SystemPrompt;
            string user = DecisionSchemaCompiler.Describe(request.State);
            string rendered = DecisionPrompt.Render(template, "diffusion-gemma", system, user);
            // The template renders an empty BOS here - the tokenizer prepends it - and a literal one in HF.
            Assert.Equal(expected.GetValue<string>(), "<bos>" + rendered);
            checkedPrompts++;
        }
        Assert.True(checkedPrompts >= 10, $"only {checkedPrompts} prompts in the fixture");
    }

    [Fact]
    public void RespellsTheSystemTurnAsAContentPart()
    {
        Assert.Equal("<|turn>system\nrules <turn|>\n<|turn>user\nhi<turn|>\n",
            DecisionPrompt.AsContentParts("<|turn>system\nrules<turn|>\n<|turn>user\nhi<turn|>\n", "  rules\n"));
        // A render that does not have the shape is left alone rather than guessed at.
        Assert.Equal("<|im_start|>system\nrules<|im_end|>", DecisionPrompt.AsContentParts("<|im_start|>system\nrules<|im_end|>", "rules"));
    }

    [ModelFact(EnvModelDir, GgufPattern)]
    public void CompilesAndTokenizesExactlyWhatDjevSends()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir)!, GgufPattern)
            ?? Environment.GetEnvironmentVariable(EnvModelDir)!;
        using DiffusionAgent agent = DiffusionAgent.Load(path, TestGates.PreferredTestBackend);
        var jevbench = Environment.GetEnvironmentVariable("TS_JEVBENCH_DIR");
        var byId = jevbench is { Length: > 0 } && Directory.Exists(jevbench)
            ? TensorSharp.Structured.Evals.JevBenchSet.Load(jevbench).Tasks.ToDictionary(t => t.Id)
            : null;

        int prompts = 0, templates = 0;
        foreach (JsonNode? record in Tasks())
        {
            DecisionRequest? request = record!["request"] is not null ? RequestOf(record)
                : byId?[record["id"]!.GetValue<string>()].ToRequest();
            if (request is null) continue;

            CompiledDecisionSchema schema = agent.Compile(request.Questions);
            string id = record["id"]!.GetValue<string>();
            Assert.True(record["template"]!.AsArray().Select(t => t!.GetValue<int>()).SequenceEqual(schema.Template), $"{id}: template");
            var slots = record["slots"]!.AsArray();
            Assert.Equal(slots.Count, schema.Slots.Count);
            for (int i = 0; i < slots.Count; i++)
            {
                Assert.Equal(slots[i]!["position"]!.GetValue<int>(), schema.Slots[i].Position);
                Assert.Equal(slots[i]!["token_ids"]!.AsArray().Select(t => t!.GetValue<int>()), schema.Slots[i].TokenIds);
            }
            Assert.Equal(record["width"]!.GetValue<int>(), schema.CanvasWidth);
            Assert.Equal(record["canvas_seed0"]!.AsArray().Select(t => t!.GetValue<int>()),
                DecisionSchemaCompiler.SeedCanvas(schema, 0, agent.Reader.VocabSize));
            templates++;

            int[] prompt = agent.Reader.EncodeChat(schema.SystemPrompt, DecisionSchemaCompiler.Describe(request.State));
            string sha = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
                "[" + string.Join(", ", prompt) + "]"))).ToLowerInvariant();
            if (sha != record["prompt_sha256"]!.GetValue<string>() && record["prompt"] is JsonArray expected)
            {
                var want = expected.Select(t => t!.GetValue<int>()).ToArray();
                int at = Enumerable.Range(0, Math.Min(want.Length, prompt.Length)).FirstOrDefault(i => want[i] != prompt[i], -1);
                _output.WriteLine($"{id}: first difference at {at} of {want.Length}/{prompt.Length}");
            }
            Assert.Equal(record["prompt_sha256"]!.GetValue<string>(), sha);
            prompts++;
        }
        _output.WriteLine($"{templates} templates and {prompts} prompts identical to djev's");
        Assert.True(templates >= 10);
    }
}
