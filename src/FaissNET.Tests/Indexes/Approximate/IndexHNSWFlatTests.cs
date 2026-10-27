using Faiss.Cpu.Indexes.Approximate;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Approximate;

public class IndexHNSWFlatTests
{
    [Theory]
    [InlineData(8, 32)]
    [InlineData(12, 4)]
    [InlineData(16, 48)]
    public void Constructor_UsesDimensionsAndMInTheirOwnSlots(int dimensions, int m)
    {
        using var hnsw = new IndexHNSW(dimensions, m);
        using var flat = new IndexHNSWFlat(dimensions, m);

        Assert.Equal(dimensions, hnsw.Dimensions);
        Assert.Equal(dimensions, flat.Dimensions);
        Assert.Equal(0, hnsw.TotalCount);
        Assert.Equal(0, flat.TotalCount);
    }

    [Theory]
    [InlineData(MetricType.L2)]
    [InlineData(MetricType.InnerProduct)]
    public void Constructor_HonoursTheRequestedMetric(MetricType metric)
    {
        using var hnsw = new IndexHNSW(8, m: 16, metric);
        using var flat = new IndexHNSWFlat(8, m: 16, metric);

        Assert.Equal(metric, hnsw.Metric);
        Assert.Equal(metric, flat.Metric);
    }

    [Fact]
    public void BaseIndex_AddAndSelfQuery_FindsEveryVector()
    {
        const int dimensions = 8;
        const int count = 32;

        using var index = new IndexHNSW(dimensions, m: 16);
        var vectors = Vectors.Random(seed: 701, count: count, dimensions: dimensions);
        index.Add(count, vectors);

        Assert.Equal(count, index.TotalCount);

        var distances = new float[1];
        var labels = new long[1];

        for (var i = 0; i < count; i++)
        {
            index.Search(1, vectors.AsSpan(i * dimensions, dimensions), k: 1, distances, labels);

            Assert.Equal(i, labels[0]);
            FloatAssert.Equal(0f, distances[0]);
        }
    }
}
