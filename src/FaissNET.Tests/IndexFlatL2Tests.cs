namespace Faiss.Tests;

using Cpu;
using Xunit;

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
}