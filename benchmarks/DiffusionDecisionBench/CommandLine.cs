// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System.Globalization;

namespace DiffusionDecisionBench;

/// <summary>`--name value`, `--name=value` and bare `--flag`; repeated names keep every value.</summary>
internal sealed class CommandLine
{
    private readonly Dictionary<string, List<string>> _values = new(StringComparer.Ordinal);

    public List<string> Positional { get; } = new();

    public static CommandLine Parse(string[] args)
    {
        var line = new CommandLine();
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (!arg.StartsWith("--", StringComparison.Ordinal)) { line.Positional.Add(arg); continue; }
            string name = arg[2..];
            string? value = null;
            int eq = name.IndexOf('=');
            if (eq >= 0) { value = name[(eq + 1)..]; name = name[..eq]; }
            else if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)) value = args[++i];
            if (!line._values.TryGetValue(name, out var list)) line._values[name] = list = new List<string>();
            if (value is not null) list.Add(value);
        }
        return line;
    }

    public bool Has(string name) => _values.ContainsKey(name);

    public string? Value(string name) => _values.TryGetValue(name, out var list) && list.Count > 0 ? list[^1] : null;

    public IReadOnlyList<string> Values(string name) => _values.TryGetValue(name, out var list) ? list : Array.Empty<string>();

    public int? Int(string name) => Value(name) is string s ? int.Parse(s, CultureInfo.InvariantCulture) : null;

    public double? Double(string name) => Value(name) is string s ? double.Parse(s, CultureInfo.InvariantCulture) : null;
}
