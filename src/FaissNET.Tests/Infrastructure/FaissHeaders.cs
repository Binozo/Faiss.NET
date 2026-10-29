using System.Globalization;
using System.Text.RegularExpressions;

namespace Faiss.Tests.Infrastructure;

/// <summary>
/// Reads constants out of the vendored faiss C++ headers.
/// </summary>
public static class FaissHeaders
{
    public const string Trait = "VendoredFaiss";

    private static readonly string Root = FindVendoredFaiss();

    /// <summary>Value of a <c>const int NAME = …;</c> declaration, resolving references to earlier constants.</summary>
    public static int IntConstant(string headerRelativePath, string name)
    {
        var constants = IntConstants(headerRelativePath);

        Assert(constants.TryGetValue(name, out var value), $"'{name}' is not declared in {headerRelativePath}. Did faiss rename it?");

        return value;
    }

    /// <summary>Ordinal of a member in an unvalued C++ <c>enum</c>, as the compiler would assign it.</summary>
    public static int EnumOrdinal(string headerRelativePath, string enumName, string member)
    {
        var text = ReadHeader(headerRelativePath);
        var declaration = Regex.Match(text, $@"enum\s+{Regex.Escape(enumName)}\s*\{{(?<body>[^}}]*)\}}");

        Assert(declaration.Success, $"enum {enumName} is not declared in {headerRelativePath}.");

        // Comments must go first: faiss documents members with /// text that itself contains commas,
        // e.g. "[min - rs*(max-min), max + rs*(max-min)]".
        var members = StripComments(declaration.Groups["body"].Value)
            .Split(',')
            .Select(m => m.Trim())
            .Where(m => m.Length > 0)
            .ToArray();

        for (var i = 0; i < members.Length; i++)
        {
            Assert(!members[i].Contains('='), $"enum {enumName} member '{members[i]}' carries an explicit value; this helper assumes implicit ordinals.");

            if (members[i] == member)
            {
                return i;
            }
        }

        Assert(false, $"enum {enumName} in {headerRelativePath} has no member '{member}'.");
        return -1;
    }

    private static Dictionary<string, int> IntConstants(string headerRelativePath)
    {
        var resolved = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (Match match in Regex.Matches(ReadHeader(headerRelativePath), @"^\s*const\s+int\s+(?<name>\w+)\s*=\s*(?<expr>[^;]+);", RegexOptions.Multiline))
        {
            resolved[match.Groups["name"].Value] = Evaluate(match.Groups["expr"].Value, resolved);
        }

        return resolved;
    }

    /// <summary>Evaluates the <c>|</c> and <c>&lt;&lt;</c> expressions faiss uses for its IO flags.</summary>
    private static int Evaluate(string expression, Dictionary<string, int> known)
    {
        var value = 0;

        foreach (var term in StripComments(expression).Split('|'))
        {
            var shifted = term.Split("<<");
            var result = Term(shifted[0], known);

            for (var i = 1; i < shifted.Length; i++)
            {
                result <<= Term(shifted[i], known);
            }

            value |= result;
        }

        return value;
    }

    private static int Term(string token, Dictionary<string, int> known)
    {
        token = token.Trim();

        if (known.TryGetValue(token, out var referenced))
        {
            return referenced;
        }

        if (token.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
        {
            return int.Parse(token[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture);
        }

        Assert(int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var literal), $"Cannot evaluate header term '{token}'.");

        return literal;
    }

    private static string StripComments(string text) =>
        Regex.Replace(Regex.Replace(text, @"/\*.*?\*/", string.Empty, RegexOptions.Singleline), @"//[^\n]*", string.Empty);

    private static string ReadHeader(string headerRelativePath)
    {
        var path = Path.Combine(Root, headerRelativePath);

        Assert(File.Exists(path), $"Vendored header not found at '{path}'.");

        return File.ReadAllText(path);
    }

    /// <summary>Walks up from the test binaries to the repository's vendored faiss checkout.</summary>
    private static string FindVendoredFaiss()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            var candidate = Path.Combine(directory.FullName, "faiss");

            if (File.Exists(Path.Combine(candidate, "faiss", "index_io.h")))
            {
                return candidate;
            }
        }

        throw new InvalidOperationException($"No vendored faiss checkout above '{AppContext.BaseDirectory}'. These tests read the real C++ headers; a literal would only restate the managed value.");
    }

    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }
}
