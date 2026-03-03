namespace Faiss.Tests;

using Cpu;
using Xunit;

/// <summary>
/// Tests the core Add and Search functionality for dot product calculations.
/// </summary>
public class IndexFlatIPTests
{
    [Fact]
    public void AddAndSearch_CalculatesCorrectInnerProduct()
    {
        int dimensions = 2;
        using var index = new IndexFlatIP(dimensions);

        Assert.Equal(dimensions, index.Dimensions);
        Assert.Equal(0, index.TotalCount);

        float[] vectors = { 2.0f, 3.0f };

        index.Add(1, vectors);

        Assert.Equal(1, index.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];

        index.Search(1, vectors, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(13.0f, distances[0]);
    }
}