using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Extensions;

public class BinaryIndexExtensionsTests
{
    private const int Bits = 16;
    private const int CodeSize = Bits / 8;
    private const int Count = 8;

    [Fact]
    public void AddSpan_InfersVectorCountFromCodeSizeNotBits()
    {
        using var index = new IndexBinaryFlat(Bits);
        var vectors = Vectors.RandomBinary(seed: 200, count: Count, bitDimensions: Bits);

        Assert.Equal(Count * CodeSize, vectors.Length);

        index.Add(vectors);

        Assert.Equal(Count, index.TotalCount);
    }

    [Fact]
    public void AddSpan_WiderVectorsStillCountCorrectly()
    {
        const int bits = 64;
        const int codeSize = bits / 8;

        using var index = new IndexBinaryFlat(bits);
        var vectors = Vectors.RandomBinary(seed: 201, count: 4, bitDimensions: bits);

        Assert.Equal(4 * codeSize, vectors.Length);

        index.Add(vectors);

        Assert.Equal(4, index.TotalCount);
    }

    [Fact]
    public void AddSpan_LengthNotMultipleOfCodeSize_Throws()
    {
        using var index = new IndexBinaryFlat(Bits);

        Assert.Throws<ArgumentException>(() => index.Add(new byte[CodeSize + 1]));
        Assert.Throws<ArgumentException>(() => index.Add(ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void AddMemoryList_FlattensChunksInOrder()
    {
        using var index = new IndexBinaryFlat(Bits);
        var vectors = Vectors.RandomBinary(seed: 202, count: Count, bitDimensions: Bits);

        var chunks = new List<ReadOnlyMemory<byte>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<byte>(vectors, i * CodeSize, CodeSize));
        }

        index.Add(chunks);

        Assert.Equal(Count, index.TotalCount);

        for (var i = 0; i < Count; i++)
        {
            Assert.Equal(vectors.AsSpan(i * CodeSize, CodeSize).ToArray(), index.Reconstruct(i));
        }
    }

    [Fact]
    public void AddMemoryList_WrongChunkSize_Throws()
    {
        using var index = new IndexBinaryFlat(Bits);

        var chunks = new List<ReadOnlyMemory<byte>>
        {
            new byte[CodeSize],
            new byte[CodeSize + 1],
        };

        Assert.Throws<ArgumentException>(() => index.Add(chunks));
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMemoryList_Empty_IsNoOp()
    {
        using var index = new IndexBinaryFlat(Bits);

        index.Add(new List<ReadOnlyMemory<byte>>());

        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AssignSpan_MatchesLowLevelAssign()
    {
        const int k = 2;

        using var index = new IndexBinaryFlat(Bits);
        var vectors = Vectors.RandomBinary(seed: 203, count: Count, bitDimensions: Bits);
        index.Add(Count, vectors);

        var expected = new long[Count * k];
        index.Assign(Count, vectors, k, expected);

        using var assigned = index.Assign(vectors, k);

        Assert.Equal(Count, assigned.QueriesCount);
        Assert.Equal(k, assigned.K);
        Assert.Equal(expected, assigned.Labels.ToArray());
    }

    [Fact]
    public void AssignView_WritesIntoCallerBuffer()
    {
        const int k = 2;

        using var index = new IndexBinaryFlat(Bits);
        var vectors = Vectors.RandomBinary(seed: 204, count: Count, bitDimensions: Bits);
        index.Add(Count, vectors);

        Span<long> labels = stackalloc long[k];
        index.Assign(vectors.AsSpan(0, CodeSize), k, labels);

        Assert.Equal(0, labels[0]);
        Assert.InRange(labels[1], 0L, Count - 1L);
    }

    [Fact]
    public void AssignView_InvalidArguments_Throw()
    {
        using var index = new IndexBinaryFlat(Bits);
        index.Add(Count, Vectors.RandomBinary(seed: 205, count: Count, bitDimensions: Bits));

        Assert.Throws<ArgumentException>(() => index.Assign(new byte[CodeSize + 1], 1, new long[1]));
        Assert.Throws<ArgumentException>(() => index.Assign(new byte[CodeSize], 2, new long[1]));
    }

    [Fact]
    public async Task TrainAsync_MemoryList_TrainsTheIndex()
    {
        using var index = new IndexBinaryIVF(Bits, nlist: 2, hnswM: null);
        Assert.False(index.IsTrained);

        var train = Vectors.RandomBinary(seed: 206, count: 64, bitDimensions: Bits);
        var chunks = new List<ReadOnlyMemory<byte>>();
        for (var i = 0; i < 64; i++)
        {
            chunks.Add(new ReadOnlyMemory<byte>(train, i * CodeSize, CodeSize));
        }

        await index.TrainAsync(chunks);

        Assert.True(index.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_WrongChunkSize_Throws()
    {
        using var index = new IndexBinaryIVF(Bits, nlist: 2, hnswM: null);

        var chunks = new List<ReadOnlyMemory<byte>> { new byte[CodeSize + 1] };

        await Assert.ThrowsAsync<ArgumentException>(() => index.TrainAsync(chunks));
        Assert.False(index.IsTrained);
    }

    [Fact]
    public void AddMemoryList_BadChunk_AddsNothing()
    {
        using var index = new IndexBinaryFlat(Bits);
        var chunks = new List<ReadOnlyMemory<byte>> { new byte[CodeSize], new byte[CodeSize - 1] };

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.Add(chunks)).ParamName);
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void AssignSpan_NonPositiveK_NamesK(int k)
    {
        using var index = new IndexBinaryFlat(Bits);
        index.Add(Vectors.RandomBinary(seed: 30, count: Count, bitDimensions: Bits));

        Assert.Equal("k", Assert.Throws<ArgumentOutOfRangeException>(() => { using var r = index.Assign(new byte[CodeSize], k); }).ParamName);
    }

    [Fact]
    public void InvalidLengths_NameTheArgument()
    {
        using var index = new IndexBinaryFlat(Bits);
        index.Add(Vectors.RandomBinary(seed: 31, count: Count, bitDimensions: Bits));

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.Add(new byte[CodeSize + 1])).ParamName);
        Assert.Equal("queryVectors", Assert.Throws<ArgumentException>(() => { using var r = index.Assign(new byte[CodeSize + 1], 1); }).ParamName);
        Assert.Equal("queryVector", Assert.Throws<ArgumentException>(() => index.Assign(new byte[CodeSize + 1], 1, new long[1])).ParamName);
        Assert.Equal("labels", Assert.Throws<ArgumentException>(() => index.Assign(new byte[CodeSize], 2, new long[1])).ParamName);
    }
}
