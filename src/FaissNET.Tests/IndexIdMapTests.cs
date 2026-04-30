using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests for ID-mapped indexes.
/// </summary>
public class IndexIdMapTests
{
    [Fact]
    public void AddWithIds_AndSearch_ReturnsCustomDatabaseIds()
    {
        int dimensions = 4;

        using var innerIndex = new IndexFlatL2(dimensions);
        using var idMapIndex = new IndexIDMap<IndexFlatL2>(innerIndex);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        long[] customIds = { 42069, 1337 };

        idMapIndex.Add(2, vectors, customIds);
        Assert.Equal(2, idMapIndex.TotalCount);

        float[] query = { 5.0f, 6.0f, 7.0f, 8.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        idMapIndex.Search(1, query, 1, distances, labels);

        Assert.Equal(1337, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}
