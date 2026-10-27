using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.IVF;

public class IVFCodecTests
{
    private const int Dimensions = 8;

    public sealed record CodecCase(string Name, int Nlist, Func<int, ICodeFloatIndex> Create)
    {
        public override string ToString() => $"{Name} nlist={Nlist}";
    }

    public static TheoryData<CodecCase> Cases { get; } =
    [
        new CodecCase("IVFPQ", 2, nlist => new IndexIVFPQ(Dimensions, n: nlist, m: 4)),
        new CodecCase("IVFSQ", 2, nlist => new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist, QuantizerType.QT_8bit, ownQuantizer: true)),
        // 300 lists need a two-byte prefix.
        new CodecCase("IVFPQ", 300, nlist => new IndexIVFPQ(Dimensions, n: nlist, m: 4)),
        new CodecCase("IVFSQ", 300, nlist => new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist, QuantizerType.QT_8bit, ownQuantizer: true)),
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Decode_ListNumberBeyondNlist_ThrowsInsteadOfAborting(CodecCase c)
    {
        using var disposable = (IDisposable)c.Create(c.Nlist);
        var (index, codes, codeSize) = await EncodeTwo((ICodeFloatIndex)disposable);

        WriteListNumber(codes.AsSpan(codeSize), c.Nlist, c.Nlist);

        Assert.Throws<ArgumentException>(() => index.Decode(2, codes, new float[2 * Dimensions]));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Decode_LastValidListNumber_Decodes(CodecCase c)
    {
        using var disposable = (IDisposable)c.Create(c.Nlist);
        var (index, codes, codeSize) = await EncodeTwo((ICodeFloatIndex)disposable);

        WriteListNumber(codes.AsSpan(codeSize), c.Nlist - 1, c.Nlist);

        var decoded = new float[2 * Dimensions];
        index.Decode(2, codes, decoded);
        Assert.All(decoded, v => Assert.True(float.IsFinite(v)));
    }

    private static async Task<(ICodeFloatIndex Index, byte[] Codes, int CodeSize)> EncodeTwo(ICodeFloatIndex index)
    {
        await ((ITrainableFloatIndex)index).TrainAsync(512, Vectors.Random(seed: 870, count: 512, dimensions: Dimensions));
        var codeSize = (int)index.GetStandaloneCodeSize();
        var codes = new byte[2 * codeSize];
        index.Encode(2, Vectors.Random(seed: 871, count: 2, dimensions: Dimensions), codes);
        return (index, codes, codeSize);
    }

    /// <summary>Writes the list number little-endian over the prefix, which is as wide as nlist - 1 needs.</summary>
    private static void WriteListNumber(Span<byte> code, int listNumber, int nlist)
    {
        var b = 0;
        for (var remaining = nlist - 1; remaining > 0; remaining >>= 8, b++)
        {
            code[b] = (byte)(listNumber >> (8 * b));
        }
    }
}
