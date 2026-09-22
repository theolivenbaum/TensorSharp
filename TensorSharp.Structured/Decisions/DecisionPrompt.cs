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
using TensorSharp.Runtime;

namespace TensorSharp.Structured.Decisions
{
    /// <summary>
    /// The chat prompt djev sends: <c>[system, user]</c> through the model's own template, a generation
    /// prompt, thinking off.
    ///
    /// djev hands the template <i>content parts</i> (<c>[{"type": "text", "text": ...}]</c>), because that is
    /// what vLLM renders, and Gemma 4's template does not spell the two forms alike: a system <i>string</i> is
    /// written as <c>trim(text)</c>, a system <i>text part</i> as <c>trim(text) + ' '</c>. So djev's prompts
    /// carry a space before the system turn's <c>&lt;turn|&gt;</c> that a string-content render does not. A
    /// chat message here is a string, so the render is corrected to the content-part spelling; the user
    /// turn is <c>trim(text)</c> either way.
    /// </summary>
    public static class DecisionPrompt
    {
        private static readonly IPromptRenderer Renderer = new GgufPromptRenderer();

        /// <summary>The prompt text for <paramref name="system"/> and <paramref name="user"/>, before
        /// tokenization (the template renders an empty BOS; the tokenizer adds it).</summary>
        public static string Render(string chatTemplate, string architecture, string system, string user)
        {
            var messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = system },
                new() { Role = "user", Content = user },
            };
            string rendered = Renderer.Render(chatTemplate, messages,
                addGenerationPrompt: true, architecture: architecture, enableThinking: false);
            return AsContentParts(rendered, system);
        }

        /// <summary>Respell a string-content system turn as the content-part one djev renders.</summary>
        internal static string AsContentParts(string rendered, string system)
        {
            string head = "<|turn>system\n" + system.Trim();
            int at = rendered.IndexOf(head + "<turn|>", StringComparison.Ordinal);
            return at < 0 ? rendered : rendered.Insert(at + head.Length, " ");
        }
    }
}
