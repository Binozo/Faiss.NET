using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Exceptions;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Factory;

public class IndexFactoryTests
{
    [Fact]
    public void Create_HnswFlat_AddAndSelfQuery()
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("HNSW16,Flat", 4, MetricType.L2);
        Assert.Equal(4, index.Dimensions);

        var vectors = Vectors.Random(seed: 1, count: 8, dimensions: 4);
        index.Add(8, vectors);

        var query = vectors.AsSpan(0, 4);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        index.Search(1, query, k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Fact]
    public async Task Create_IvfFlat_TrainAddAndSelfQuery()
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("IVF1,Flat", 4, MetricType.L2);
        Assert.False(index.IsTrained);

        var train = Vectors.Random(seed: 1, count: 40, dimensions: 4);
        await index.TrainAsync(40, train);
        Assert.True(index.IsTrained);

        var added = Vectors.Random(seed: 2, count: 4, dimensions: 4);
        index.Add(4, added);

        var query = added.AsSpan(0, 4);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        index.Search(1, query, k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Theory]
    [InlineData("Flat")]
    [InlineData("IVF4,Flat")]
    [InlineData("IVF4,PQ2")]
    [InlineData("HNSW16,Flat")]
    [InlineData("LSH")]
    [InlineData("PCA2,Flat")]
    public void Create_BuildsIndexWithRequestedShape(string description)
    {
        using var index = IndexFactory.Create<GenericFloatIndex>(description, 8, MetricType.L2);

        Assert.Equal(8, index.Dimensions);
        Assert.Equal(0, index.TotalCount);
        Assert.Equal(MetricType.L2, index.Metric);
    }

    [Fact]
    public void Create_IdMapPrefix_AcceptsCustomIds()
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("IDMap,Flat", 4, MetricType.L2);

        var vectors = Vectors.Random(seed: 3, count: 3, dimensions: 4);
        index.Add(3, vectors, [70L, 80L, 90L]);

        Assert.Equal(3, index.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(4, 4), 1, distances, labels);

        Assert.Equal(80, labels[0]);
    }

    [Fact]
    public void Create_HonoursTheRequestedMetric()
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("Flat", 4, MetricType.InnerProduct);

        Assert.Equal(MetricType.InnerProduct, index.Metric);
    }

    [Fact]
    public void Create_UnknownDescription_Throws()
    {
        Assert.Throws<FaissNativeException>(() => IndexFactory.Create<GenericFloatIndex>("NotAnIndex", 4, MetricType.L2));
    }

    [Fact]
    public void Create_ConcreteType_IsUsableThroughItsOwnApi()
    {
        using var index = IndexFactory.Create<IndexFlatL2>("Flat", 4);

        var vectors = Vectors.Random(seed: 4, count: 2, dimensions: 4);
        index.Add(2, vectors);

        FloatAssert.Equal(vectors.AsSpan(0, 4), index.Reconstruct(0));
    }
}
