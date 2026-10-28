using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class CloneContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.ClonableFloat), MemberType = typeof(IndexCatalog))]
    public void Clone_IsIndependentCopy(IndexCase c)
    {
        Assert.NotNull(c.CloneIndex);

        using var original = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        Sequential(original).Add(4, vectors);

        using var clone = c.CloneIndex(original);

        Assert.Equal(original.Dimensions, clone.Dimensions);
        Assert.Equal(original.Metric, clone.Metric);
        Assert.Equal(4, clone.TotalCount);

        var query = vectors.AsSpan(0, c.Dimensions);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        clone.Search(1, query, k: 1, distances, labels);
        Assert.Equal(0, labels[0]);

        Sequential(clone).Add(1, Vectors.Random(seed: 2, count: 1, dimensions: c.Dimensions));
        Assert.Equal(4, original.TotalCount);
        Assert.Equal(5, clone.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ClonableBinary), MemberType = typeof(IndexCatalog))]
    public void Clone_IsIndependentCopy_Binary(BinaryIndexCase c)
    {
        Assert.NotNull(c.CloneIndex);

        using var original = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 1, count: 4, bitDimensions: c.Dimensions);
        Sequential(original).Add(4, vectors);

        using var clone = c.CloneIndex(original);

        Assert.Equal(original.Dimensions, clone.Dimensions);
        Assert.Equal(4, clone.TotalCount);

        var query = vectors.AsSpan(0, c.CodeSize);
        Span<int> distances = stackalloc int[1];
        Span<long> labels = stackalloc long[1];
        clone.Search(1, query, k: 1, distances, labels);
        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);

        Sequential(clone).Add(1, Vectors.RandomBinary(seed: 2, count: 1, bitDimensions: c.Dimensions));
        Assert.Equal(4, original.TotalCount);
        Assert.Equal(5, clone.TotalCount);
    }

    private static IIDSequentialFloatIndex Sequential(FloatIndex index) => Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index);

    private static IIDSequentialBinaryIndex Sequential(BinaryIndex index) => Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index);
}
