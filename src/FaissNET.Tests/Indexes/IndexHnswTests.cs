using Faiss.Cpu.Indexes.Approximate;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests for HNSW graph-based approximate indexes.
/// </summary>
public class IndexHnswTests
{
    [Fact]
    public void HNSW_AddAndSearch_ReturnsValidResults()
    {
        int dimensions = 4;
        using var index = new IndexHNSW(dimensions, m: 32);

        float[] vectors = {
            1.0f, 1.0f, 1.0f, 1.0f,
            10.0f, 10.0f, 10.0f, 10.0f,
            100.0f, 100.0f, 100.0f, 100.0f
        };

        index.Add(3, vectors);
        Assert.Equal(3, index.TotalCount);

        float[] query = { 11.0f, 11.0f, 11.0f, 11.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        index.Search(1, query, 1, distances, labels);

        Assert.Equal(1, labels[0]);
        Assert.Equal(4.0f, distances[0]);
    }

    [Fact]
    public void HNSW_Reset_ClearsTheGraph()
    {
        using var index = new IndexHNSW(128);
        index.Add(1, new float[128]);

        index.Reset();

        Assert.Equal(0, index.TotalCount);
    }
}
