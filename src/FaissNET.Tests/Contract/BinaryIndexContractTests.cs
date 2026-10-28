using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class BinaryIndexContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void Create_IsEmptyWithExpectedShape(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();

        Assert.Equal(c.Dimensions, index.Dimensions);
        Assert.Equal(c.Metric, index.Metric);
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialBinary), MemberType = typeof(IndexCatalog))]
    public void Add_IncrementsTotalCount(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 1, count: 8, bitDimensions: c.Dimensions);

        Sequential(index).Add(8, vectors);

        Assert.Equal(8, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void Verbose_DefaultsFalseAndRoundTrips(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();

        Assert.False(index.Verbose);

        index.Verbose = true;
        Assert.True(index.Verbose);

        index.Verbose = false;
        Assert.False(index.Verbose);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ExactBinary), MemberType = typeof(IndexCatalog))]
    public void Search_Batch_ReportsPerQueryNeighboursAndDistances(BinaryIndexCase c)
    {
        const int count = 6;
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 2, count: count, bitDimensions: c.Dimensions);
        Sequential(index).Add(count, vectors);

        var distances = new int[count * k];
        var labels = new long[count * k];

        index.Search(count, vectors, k, distances, labels);

        for (var q = 0; q < count; q++)
        {
            var query = vectors.AsSpan(q * c.CodeSize, c.CodeSize);
            var row = q * k;

            Assert.Equal(q, labels[row]);
            Assert.Equal(0, distances[row]);
            Assert.Equal(k, labels.Skip(row).Take(k).Distinct().Count());

            for (var j = 0; j < k; j++)
            {
                var label = labels[row + j];
                Assert.InRange(label, 0L, count - 1L);
                Assert.Equal(Oracles.Hamming(query, vectors.AsSpan((int)label * c.CodeSize, c.CodeSize)), distances[row + j]);

                if (j > 0)
                    Assert.True(distances[row + j] >= distances[row + j - 1], $"Query {q} neighbour {j}: {distances[row + j]} follows {distances[row + j - 1]}.");
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ExactBinary), MemberType = typeof(IndexCatalog))]
    public void Search_KExceedsTotalCount_PadsWithSentinels(BinaryIndexCase c)
    {
        const int stored = 2;
        const int k = 5;

        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 3, count: stored, bitDimensions: c.Dimensions);
        Sequential(index).Add(stored, vectors);

        var distances = new int[k];
        var labels = new long[k];

        index.Search(1, vectors.AsSpan(0, c.CodeSize), k, distances, labels);

        for (var i = 0; i < stored; i++)
        {
            Assert.InRange(labels[i], 0L, stored - 1L);
        }

        for (var i = stored; i < k; i++)
        {
            Assert.Equal(-1, labels[i]);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialBinary), MemberType = typeof(IndexCatalog))]
    public void Search_EmptyIndex_ReturnsNoNeighbours(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var query = Vectors.RandomBinary(seed: 4, count: 1, bitDimensions: c.Dimensions);

        var distances = new int[2];
        var labels = new long[2];

        index.Search(1, query, k: 2, distances, labels);

        Assert.Equal(-1, labels[0]);
        Assert.Equal(-1, labels[1]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ExactBinary), MemberType = typeof(IndexCatalog))]
    public void Assign_ReturnsNearestLabelsPerQuery(BinaryIndexCase c)
    {
        const int count = 4;
        const int k = 2;

        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 5, count: count, bitDimensions: c.Dimensions);
        Sequential(index).Add(count, vectors);

        var labels = new long[count * k];

        index.Assign(count, vectors, k, labels);

        for (var q = 0; q < count; q++)
        {
            var row = q * k;

            Assert.Equal(q, labels[row]);
            Assert.InRange(labels[row + 1], 0L, count - 1L);
            Assert.NotEqual(labels[row], labels[row + 1]);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void Search_InvalidArguments_Throw(BinaryIndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var query = Vectors.RandomBinary(seed: 6, count: queries, bitDimensions: c.Dimensions);
        var distances = new int[queries * k];
        var labels = new long[queries * k];

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(-1, query, k, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(queries, query, 0, distances, labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query.AsSpan(0, c.CodeSize), k, distances, labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query, k, distances.AsSpan(0, queries * k - 1), labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query, k, distances, labels.AsSpan(0, queries * k - 1)));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void Assign_InvalidArguments_Throw(BinaryIndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var query = Vectors.RandomBinary(seed: 6, count: queries, bitDimensions: c.Dimensions);
        var labels = new long[queries * k];

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Assign(-1, query, k, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Assign(queries, query, 0, labels));
        Assert.Throws<ArgumentException>(() => index.Assign(queries, query.AsSpan(0, c.CodeSize), k, labels));
        Assert.Throws<ArgumentException>(() => index.Assign(queries, query, k, labels.AsSpan(0, queries * k - 1)));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialBinary), MemberType = typeof(IndexCatalog))]
    public void Reset_ZerosCount(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        Sequential(index).Add(4, Vectors.RandomBinary(seed: 1, count: 4, bitDimensions: c.Dimensions));

        index.Reset();

        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialBinary), MemberType = typeof(IndexCatalog))]
    public void Add_InvalidArguments_Throw(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 7, count: 2, bitDimensions: c.Dimensions);
        var sequential = Sequential(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => sequential.Add(-1, vectors));
        Assert.Throws<ArgumentException>(() => sequential.Add(4, vectors));
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedBinary), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_SearchReturnsThoseIds(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 8, count: 4, bitDimensions: c.Dimensions);
        long[] ids = [100, 200, 300, 400];

        Mapped(index).Add(4, vectors, ids);

        Assert.Equal(4, index.TotalCount);

        var distances = new int[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, c.CodeSize), k: 1, distances, labels);

        Assert.Equal(100, labels[0]);
        Assert.Equal(0, distances[0]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedBinary), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_InvalidArguments_Throw(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 8, count: 2, bitDimensions: c.Dimensions);
        var mapped = Mapped(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => mapped.Add(-1, vectors, [1, 2]));
        Assert.Throws<ArgumentException>(() => mapped.Add(4, vectors, [1, 2, 3, 4]));
        Assert.Throws<ArgumentException>(() => mapped.Add(2, vectors, [1]));
        Assert.Equal(0, index.TotalCount);
    }

    private static IIDSequentialBinaryIndex Sequential(BinaryIndex index) => Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index);

    private static IIDMappedBinaryIndex Mapped(BinaryIndex index) => Assert.IsAssignableFrom<IIDMappedBinaryIndex>(index);
}
