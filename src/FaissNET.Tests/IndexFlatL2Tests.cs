using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests the core Add and Search functionality for Euclidean distance.
/// </summary>
public class IndexFlatL2Tests
{
    [Fact]
    public void AddAndSearch_ReturnsExactMatch_WithZeroDistance()
    {
        int dimensions = 4;
        using var index = new IndexFlatL2(dimensions);

        Assert.Equal(dimensions, index.Dimensions);
        Assert.Equal(0, index.TotalCount);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f };

        index.Add(1, vectors);
        Assert.Equal(1, index.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];

        index.Search(1, vectors, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }

    [Fact]
    public void Search_WithOwnership_ReturnsExactMatch()
    {
        using var index = new IndexFlatL2(4);
        ReadOnlySpan<float> vectors = [1.0f, 2.0f, 3.0f, 4.0f];

        index.Add(vectors);

        using var result = index.Search(vectors, k: 1);

        Assert.Equal(0, result[0].Label);
        Assert.Equal(0.0f, result[0].Distance);
    }

    [Fact]
    public void Search_WithCallerBuffers_ZeroAllocation()
    {
        using var index = new IndexFlatL2(4);
        index.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];

        for (int i = 0; i < 10; i++)
        {
            var result = index.Search([1.0f, 2.0f, 3.0f, 4.0f], k: 1, distances, labels);
            Assert.Equal(0, result[0].Label);
        }
    }
}
