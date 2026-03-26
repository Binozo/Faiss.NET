namespace Faiss.Tests;

using Xunit;

using Cpu.Indexes;

/// <summary>
/// Proving our index IDs wrapper returns the right database IDs.
/// </summary>
public class FaissIndexIdMapTests
{
    [Fact]
    public void AddWithIds_AndSearch_ReturnsCustomDatabaseIds()
    {
        // Arrange: Spin up the basic index
        int dimensions = 4;
        
        // Note: The user owns this inner index and must dispose it!
        using var innerIndex = new FaissIndexFlatL2(dimensions);
        
        // Wrap it in our goated Decorator
        using var idMapIndex = new FaissIndexIDMap(innerIndex);

        // Create 2 vectors: [1,2,3,4] and [5,6,7,8]
        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f,   5.0f, 6.0f, 7.0f, 8.0f };
        
        // Here is the magic: Our custom database IDs!
        long[] customIds = { 42069, 1337 };

        // Act: Add them using our new interface method
        idMapIndex.AddWithIds(2, vectors, customIds);

        // Assert it actually tracked them
        Assert.Equal(2, idMapIndex.TotalCount);

        // Search for the second vector exactly
        float[] query = { 5.0f, 6.0f, 7.0f, 8.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        idMapIndex.Search(1, query, 1, distances, labels);

        // Assert: If it was a standard index, this would normally return ID 1.
        // But because of IDMap it must return our exact database ID: 1337!
        Assert.Equal(1337, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}