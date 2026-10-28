using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class CodeContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void StandaloneCodeSize_IsPositive(IndexCase c)
    {
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);

        Assert.True(codec.GetStandaloneCodeSize() > 0);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    public void EncodeDecode_Roundtrips(IndexCase c)
    {
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);
        var codeSize = (int)codec.GetStandaloneCodeSize();
        var codes = new byte[2 * codeSize];
        var decoded = new float[2 * c.Dimensions];

        codec.Encode(2, vectors, codes);
        codec.Decode(2, codes, decoded);

        FloatAssert.Equal(vectors, decoded);
    }

    /// <summary>
    /// A quantizing codec cannot return the input, but sa_decode(sa_encode(x)) must be exactly what the index itself
    /// reconstructs for a stored x: both decode the same code.
    /// </summary>
    [Theory]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void LossyEncodeDecode_MatchesTheIndexsOwnReconstruction(IndexCase c)
    {
        const int count = 16;
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var vectors = Vectors.Random(seed: 2, count: count, dimensions: c.Dimensions);

        var decoded = Roundtrip(codec, vectors, count);

        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(count, vectors);
        c.EnableReconstruct(index);
        Assert.Equal(Assert.IsAssignableFrom<IReconstructFloatIndex>(index).Reconstruct(0, count), decoded);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.NearestLossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void LossyEncodeDecode_IsNearestToItsOwnSource(IndexCase c)
    {
        const int count = 16;
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var vectors = Vectors.Random(seed: 3, count: count, dimensions: c.Dimensions);
        var d = c.Dimensions;

        var decoded = Roundtrip(codec, vectors, count);

        for (var i = 0; i < count; i++)
        {
            var own = Oracles.L2Sqr(decoded.AsSpan(i * d, d), vectors.AsSpan(i * d, d));
            for (var j = 0; j < count; j++)
            {
                Assert.True(j == i || own < Oracles.L2Sqr(decoded.AsSpan(i * d, d), vectors.AsSpan(j * d, d)),
                    $"{c.Name}: decoded vector {i} is nearer to source {j}.");
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void Codes_ArePackedAtTheStandaloneCodeSize(IndexCase c)
    {
        const int count = 3;
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var codeSize = (int)codec.GetStandaloneCodeSize();
        var vectors = Vectors.Random(seed: 4, count: count, dimensions: c.Dimensions);

        var codes = new byte[count * codeSize];
        codec.Encode(count, vectors, codes);
        var batch = new float[count * c.Dimensions];
        codec.Decode(count, codes, batch);

        for (var i = 0; i < count; i++)
        {
            var single = new float[c.Dimensions];
            codec.Decode(1, codes.AsSpan(i * codeSize, codeSize), single);
            FloatAssert.Equal(batch.AsSpan(i * c.Dimensions, c.Dimensions), single);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void Decode_UndersizedSpansOrNegativeCount_Throw(IndexCase c)
    {
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var codeSize = (int)codec.GetStandaloneCodeSize();

        Assert.Throws<ArgumentException>(() => codec.Decode(2, new byte[(2 * codeSize) - 1], new float[2 * c.Dimensions]));
        Assert.Throws<ArgumentException>(() => codec.Decode(2, new byte[2 * codeSize], new float[(2 * c.Dimensions) - 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => codec.Decode(-1, [], []));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void Encode_OutputTooSmall_Throws(IndexCase c)
    {
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);
        var vectors = Vectors.Random(seed: 1, count: 1, dimensions: c.Dimensions);

        Assert.Throws<ArgumentException>(() => codec.Encode(1, vectors, []));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.CodeFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyCodeFloat), MemberType = typeof(IndexCatalog))]
    public void Encode_NegativeCount_Throws(IndexCase c)
    {
        using var index = c.CreateIndex();
        var codec = Assert.IsAssignableFrom<ICodeFloatIndex>(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => codec.Encode(-1, [], []));
    }

    private static float[] Roundtrip(ICodeFloatIndex codec, float[] vectors, int count)
    {
        var codes = new byte[count * codec.GetStandaloneCodeSize()];
        var decoded = new float[vectors.Length];
        codec.Encode(count, vectors, codes);
        codec.Decode(count, codes, decoded);
        return decoded;
    }
}
