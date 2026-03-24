namespace Faiss.Tests;

using Xunit;

using Cpu.Indexes;

/// <summary>
/// Proving our HNSW graph index is working and lightspeed.
/// </summary>
public class FaissIndexHnswTests
{
    [Fact]
    public void HNSW_AddAndSearch_ReturnsValidResults()
    {
        // Arrange: Spin up HNSW with standard M=32 neighbors
        int dimensions = 4;
        using var index = new FaissIndexHNSW(dimensions, m: 32);

        // Create a small dataset
        float[] vectors = { 
            1.0f, 1.0f, 1.0f, 1.0f, // ID 0
            10.0f, 10.0f, 10.0f, 10.0f, // ID 1
            100.0f, 100.0f, 100.0f, 100.0f // ID 2
        };

        // Act: Build the graph
        index.Add(3, vectors);

        // Assert: Graph should have 3 nodes now
        Assert.Equal(3, index.TotalCount);

        // Search for a vector close to the second one [10, 10, 10, 10]
        float[] query = { 11.0f, 11.0f, 11.0f, 11.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        index.Search(1, query, 1, distances, labels);

        // Assert: HNSW should navigate the graph and find ID 1
        Assert.Equal(1, labels[0]);
        // Distance from [10,10,10,10] to [11,11,11,11] in L2 is 1^2 + 1^2 + 1^2 + 1^2 = 4
        Assert.Equal(4.0f, distances[0]);
    }

    [Fact]
    public void HNSW_Reset_ClearsTheGraph()
    {
        // Arrange
        using var index = new FaissIndexHNSW(128);
        index.Add(1, new float[128]);

        // Act
        index.Reset();

        // Assert
        Assert.Equal(0, index.TotalCount);
    }
}