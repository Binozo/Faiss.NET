using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class FloatIndexContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void Create_IsEmptyWithExpectedShape(IndexCase c)
    {
        using var index = c.CreateIndex();

        Assert.Equal(c.Dimensions, index.Dimensions);
        Assert.Equal(c.Metric, index.Metric);
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Add_IncrementsTotalCount(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);

        Sequential(index).Add(vectors.Length / c.Dimensions, vectors);

        Assert.Equal(8, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void Verbose_DefaultsFalseAndRoundTrips(IndexCase c)
    {
        using var index = c.CreateIndex();

        Assert.False(index.Verbose);

        index.Verbose = true;
        Assert.True(index.Verbose);

        index.Verbose = false;
        Assert.False(index.Verbose);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ExactFloat), MemberType = typeof(IndexCatalog))]
    public void Search_Batch_ReportsPerQueryNeighboursAndDistances(IndexCase c)
    {
        const int count = 6;
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 2, count: count, dimensions: c.Dimensions);
        Sequential(index).Add(count, vectors);

        var distances = new float[count * k];
        var labels = new long[count * k];

        index.Search(count, vectors, k, distances, labels);

        for (var q = 0; q < count; q++)
        {
            var query = vectors.AsSpan(q * c.Dimensions, c.Dimensions);
            var row = q * k;

            Assert.Equal(BestLabel(c, query, vectors, count), labels[row]);
            Assert.Equal(k, labels.Skip(row).Take(k).Distinct().Count());

            for (var j = 0; j < k; j++)
            {
                var label = labels[row + j];
                Assert.InRange(label, 0L, count - 1L);

                var neighbour = vectors.AsSpan((int)label * c.Dimensions, c.Dimensions);
                FloatAssert.Equal(Oracle(c.Metric, query, neighbour), distances[row + j], relTol: 1e-4f, absTol: 1e-5f);

                if (j > 0)
                {
                    Assert.True(
                        Ordered(c.Metric, distances[row + j - 1], distances[row + j]),
                        $"Query {q} neighbour {j}: {distances[row + j]:G9} out of order after {distances[row + j - 1]:G9} for {c.Metric}."
                    );
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Search_KExceedsTotalCount_PadsWithSentinels(IndexCase c)
    {
        const int stored = 2;
        const int k = 5;

        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 3, count: stored, dimensions: c.Dimensions);
        Sequential(index).Add(stored, vectors);

        var distances = new float[k];
        var labels = new long[k];

        index.Search(1, vectors.AsSpan(0, c.Dimensions), k, distances, labels);

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
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Search_EmptyIndex_ReturnsNoNeighbours(IndexCase c)
    {
        using var index = c.CreateIndex();
        var query = Vectors.Random(seed: 4, count: 1, dimensions: c.Dimensions);

        var distances = new float[2];
        var labels = new long[2];

        index.Search(1, query, k: 2, distances, labels);

        Assert.Equal(-1, labels[0]);
        Assert.Equal(-1, labels[1]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Assign_ReturnsNearestLabelsPerQuery(IndexCase c)
    {
        const int count = 4;
        const int k = 2;

        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 5, count: count, dimensions: c.Dimensions);
        Sequential(index).Add(count, vectors);

        var labels = new long[count * k];

        index.Assign(count, vectors, k, labels);

        for (var q = 0; q < count; q++)
        {
            var row = q * k;

            Assert.Equal(BestLabel(c, vectors.AsSpan(q * c.Dimensions, c.Dimensions), vectors, count), labels[row]);
            Assert.InRange(labels[row + 1], 0L, count - 1L);
            Assert.NotEqual(labels[row], labels[row + 1]);
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Reset_ZerosCount(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        Sequential(index).Add(vectors.Length / c.Dimensions, vectors);

        index.Reset();

        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedFloat), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_SearchReturnsThoseIds(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        long[] ids = [100, 200, 300, 400];

        Mapped(index).Add(4, vectors, ids);

        Assert.Equal(4, index.TotalCount);

        var query = vectors.AsSpan(0, c.Dimensions);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        index.Search(1, query, k: 1, distances, labels);

        Assert.Equal(100, labels[0]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedFloat), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_Reset_ZerosCount(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);
        long[] ids = [10, 20];

        Mapped(index).Add(2, vectors, ids);

        index.Reset();

        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedFloat), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_TooFewIds_Throws(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);
        long[] ids = [1];

        var mapped = Mapped(index);
        Assert.Throws<ArgumentException>(() => mapped.Add(2, vectors, ids));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public void Add_InvalidArguments_Throw(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 7, count: 2, dimensions: c.Dimensions);
        var sequential = Sequential(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => sequential.Add(-1, vectors));
        Assert.Throws<ArgumentException>(() => sequential.Add(4, vectors));
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.MappedFloat), MemberType = typeof(IndexCatalog))]
    public void AddWithIds_InvalidArguments_Throw(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 7, count: 2, dimensions: c.Dimensions);
        long[] ids = [1, 2, 3, 4];
        var mapped = Mapped(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => mapped.Add(-1, vectors, ids));
        Assert.Throws<ArgumentException>(() => mapped.Add(4, vectors, ids));
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void Search_InvalidArguments_Throw(IndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var query = Vectors.Random(seed: 6, count: queries, dimensions: c.Dimensions);
        var distances = new float[queries * k];
        var labels = new long[queries * k];

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(-1, query, k, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(queries, query, 0, distances, labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query.AsSpan(0, c.Dimensions), k, distances, labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query, k, distances.AsSpan(0, queries * k - 1), labels));
        Assert.Throws<ArgumentException>(() => index.Search(queries, query, k, distances, labels.AsSpan(0, queries * k - 1)));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void Assign_InvalidArguments_Throw(IndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var query = Vectors.Random(seed: 6, count: queries, dimensions: c.Dimensions);
        var labels = new long[queries * k];

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Assign(-1, query, k, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => index.Assign(queries, query, 0, labels));
        Assert.Throws<ArgumentException>(() => index.Assign(queries, query.AsSpan(0, c.Dimensions), k, labels));
        Assert.Throws<ArgumentException>(() => index.Assign(queries, query, k, labels.AsSpan(0, queries * k - 1)));
    }

    private static IIDSequentialFloatIndex Sequential(FloatIndex index) =>
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index);

    private static IIDMappedFloatIndex Mapped(FloatIndex index) =>
        Assert.IsAssignableFrom<IIDMappedFloatIndex>(index);

    private static float Oracle(MetricType metric, ReadOnlySpan<float> a, ReadOnlySpan<float> b) => metric switch
    {
        MetricType.L2 => Oracles.L2Sqr(a, b),
        MetricType.InnerProduct => Oracles.InnerProduct(a, b),
        _ => throw new NotSupportedException($"No distance oracle for {metric}."),
    };

    private static bool Ordered(MetricType metric, float previous, float current) =>
        metric == MetricType.InnerProduct ? current <= previous + 1e-5f : current >= previous - 1e-5f;

    private static long BestLabel(IndexCase c, ReadOnlySpan<float> query, float[] vectors, int count)
    {
        var best = 0L;
        var bestScore = Oracle(c.Metric, query, vectors.AsSpan(0, c.Dimensions));

        for (var i = 1; i < count; i++)
        {
            var score = Oracle(c.Metric, query, vectors.AsSpan(i * c.Dimensions, c.Dimensions));
            if (Ordered(c.Metric, bestScore, score) && Math.Abs(score - bestScore) > 1e-5f)
                continue;

            best = i;
            bestScore = score;
        }

        return best;
    }
}
