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
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TensorSharp.Structured.Decisions
{
    using JsonValue = System.Text.Json.Nodes.JsonValue;

    /// <summary>
    /// <c>json.dumps(value, ensure_ascii=False, separators=(",", ":"))</c>, byte for byte.
    ///
    /// Structured state and structured criteria reach the model as JSON text, and djev also hashes JSON
    /// to derive per-question seeds. <c>System.Text.Json</c> escapes differently (it escapes non-ASCII and
    /// HTML-sensitive characters by default) and spells floats differently (<c>1E+16</c> against
    /// Python's <c>1e+16</c>, <c>2</c> against <c>2.0</c>), and a different string is a different prompt
    /// and a different seed. So this writes CPython's spelling.
    /// </summary>
    public static class PythonJson
    {
        /// <summary>Compact <c>json.dumps</c> of a JSON tree. <paramref name="sortKeys"/> is
        /// <c>sort_keys=True</c>.</summary>
        public static string Dumps(JsonNode? node, bool sortKeys = false)
        {
            var builder = new StringBuilder();
            Write(builder, node, sortKeys);
            return builder.ToString();
        }

        /// <summary>A JSON string literal, escaped as <c>json.dumps(text, ensure_ascii=False)</c>.</summary>
        public static string Quote(string text)
        {
            var builder = new StringBuilder(text.Length + 2);
            WriteString(builder, text);
            return builder.ToString();
        }

        /// <summary>
        /// Turn a CLR value into the JSON tree it stands for: strings, numbers, booleans, <see
        /// cref="JsonNode"/>, <see cref="JsonElement"/>, dictionaries (in their enumeration order) and
        /// sequences. Key order is preserved - it is part of what the model reads.
        /// </summary>
        public static JsonNode? ToNode(object? value) => value switch
        {
            null => null,
            JsonNode node => node.DeepClone(),
            JsonElement element => element.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined
                ? null : JsonNode.Parse(element.GetRawText()),
            JsonDocument document => ToNode(document.RootElement),
            string text => JsonValue.Create(text),
            bool flag => JsonValue.Create(flag),
            byte or sbyte or short or ushort or int or uint or long => JsonValue.Create(Convert.ToInt64(value, CultureInfo.InvariantCulture)),
            ulong u => JsonValue.Create(u),
            float f => JsonValue.Create((double)f),
            double d => JsonValue.Create(d),
            decimal m => JsonValue.Create(m),
            IEnumerable<KeyValuePair<string, object?>> pairs => ObjectOf(pairs),
            IDictionary dictionary => ObjectOf(dictionary.Cast<DictionaryEntry>()
                .Select(e => new KeyValuePair<string, object?>(e.Key.ToString() ?? "", e.Value))),
            IEnumerable sequence => new JsonArray(sequence.Cast<object?>().Select(ToNode).ToArray()),
            _ => JsonValue.Create(value.ToString() ?? ""),
        };

        private static JsonObject ObjectOf(IEnumerable<KeyValuePair<string, object?>> pairs)
        {
            var result = new JsonObject();
            foreach ((string key, object? value) in pairs) result[key] = ToNode(value);
            return result;
        }

        private static void Write(StringBuilder builder, JsonNode? node, bool sortKeys)
        {
            switch (node)
            {
                case null:
                    builder.Append("null");
                    return;
                case JsonObject obj:
                {
                    builder.Append('{');
                    IEnumerable<KeyValuePair<string, JsonNode?>> properties = sortKeys
                        ? obj.OrderBy(p => p.Key, StringComparer.Ordinal)
                        : obj;
                    bool first = true;
                    foreach ((string key, JsonNode? value) in properties)
                    {
                        if (!first) builder.Append(',');
                        first = false;
                        WriteString(builder, key);
                        builder.Append(':');
                        Write(builder, value, sortKeys);
                    }
                    builder.Append('}');
                    return;
                }
                case JsonArray array:
                {
                    builder.Append('[');
                    for (int i = 0; i < array.Count; i++)
                    {
                        if (i > 0) builder.Append(',');
                        Write(builder, array[i], sortKeys);
                    }
                    builder.Append(']');
                    return;
                }
                case JsonValue value:
                    WriteValue(builder, value);
                    return;
            }
        }

        private static void WriteValue(StringBuilder builder, JsonValue value)
        {
            switch (value.GetValueKind())
            {
                case JsonValueKind.String:
                    WriteString(builder, value.GetValue<string>());
                    return;
                case JsonValueKind.True:
                    builder.Append("true");
                    return;
                case JsonValueKind.False:
                    builder.Append("false");
                    return;
                case JsonValueKind.Null:
                    builder.Append("null");
                    return;
                case JsonValueKind.Number:
                    builder.Append(Number(value));
                    return;
                default:
                    throw new FormatException($"Unsupported JSON value kind {value.GetValueKind()}.");
            }
        }

        /// <summary>
        /// A number as Python would have read it back: a literal without a fraction or exponent is an
        /// <c>int</c> and keeps its digits, anything else is a <c>float</c> and gets <c>repr</c>.
        /// </summary>
        private static string Number(JsonValue value)
        {
            if (value.TryGetValue(out JsonElement element))
                return Literal(element.GetRawText());
            if (value.TryGetValue(out double d)) return FormatFloat(d);
            if (value.TryGetValue(out float f)) return FormatFloat(f);
            if (value.TryGetValue(out decimal m)) return FormatFloat((double)m);
            return Literal(value.ToJsonString());
        }

        private static string Literal(string raw) =>
            raw.IndexOfAny(new[] { '.', 'e', 'E' }) < 0
                ? raw
                : FormatFloat(double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture));

        /// <summary>CPython's <c>float.__repr__</c>: the shortest round-tripping digits, fixed notation
        /// for decimal exponents in [-4, 16), scientific with a two-digit exponent otherwise.</summary>
        public static string FormatFloat(double value)
        {
            if (double.IsNaN(value)) return "NaN";
            if (double.IsPositiveInfinity(value)) return "Infinity";
            if (double.IsNegativeInfinity(value)) return "-Infinity";
            if (value == 0) return double.IsNegative(value) ? "-0.0" : "0.0";

            string shortest = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
            int exponent = 0;
            int e = shortest.IndexOfAny(new[] { 'E', 'e' });
            string mantissa = shortest;
            if (e >= 0)
            {
                exponent = int.Parse(shortest[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                mantissa = shortest[..e];
            }
            int point = mantissa.IndexOf('.');
            string digits = point < 0 ? mantissa : mantissa.Remove(point, 1);
            // decpt: where the decimal point sits relative to the first digit, Python's convention.
            int decpt = (point < 0 ? mantissa.Length : point) + exponent;
            int lead = 0;
            while (lead < digits.Length - 1 && digits[lead] == '0') lead++;
            digits = digits[lead..];
            decpt -= lead;
            digits = digits.TrimEnd('0');
            if (digits.Length == 0) digits = "0";

            var text = new StringBuilder();
            if (value < 0) text.Append('-');
            if (decpt is > -4 and <= 16)
            {
                if (decpt <= 0)
                    text.Append("0.").Append('0', -decpt).Append(digits);
                else if (decpt >= digits.Length)
                    text.Append(digits).Append('0', decpt - digits.Length).Append(".0");
                else
                    text.Append(digits, 0, decpt).Append('.').Append(digits, decpt, digits.Length - decpt);
            }
            else
            {
                text.Append(digits[0]);
                if (digits.Length > 1) text.Append('.').Append(digits, 1, digits.Length - 1);
                int exp = decpt - 1;
                text.Append('e').Append(exp < 0 ? '-' : '+')
                    .Append(Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture));
            }
            return text.ToString();
        }

        private static void WriteString(StringBuilder builder, string value)
        {
            builder.Append('"');
            foreach (char c in value)
            {
                switch (c)
                {
                    case '"': builder.Append("\\\""); break;
                    case '\\': builder.Append("\\\\"); break;
                    case '\n': builder.Append("\\n"); break;
                    case '\r': builder.Append("\\r"); break;
                    case '\t': builder.Append("\\t"); break;
                    case '\b': builder.Append("\\b"); break;
                    case '\f': builder.Append("\\f"); break;
                    default:
                        if (c < 0x20)
                            builder.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            builder.Append(c);
                        break;
                }
            }
            builder.Append('"');
        }
    }
}
