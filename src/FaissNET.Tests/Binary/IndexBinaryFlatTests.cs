using System;
using Faiss.Cpu.Indexes.Binary;
using Xunit;

namespace Faiss.Tests.Binary;

/// <summary>
/// Tests for binary flat indexes using Hamming distance.
/// </summary>
public class IndexBinaryFlatTests
{
    [Fact]
    public void AddAndSearch_BinaryVectors_ReturnsExactHammingDistance()
    {
        int dimensions = 16;
        using var index = new IndexBinaryFlat(dimensions);

        Assert.Equal(dimensions, index.Dimensions);
        Assert.Equal(0, index.TotalCount);

        byte[] vectors = {
            255, 0,
            0, 255
        };

        index.Add(2, vectors);
        Assert.Equal(2, index.TotalCount);

        byte[] query = { 255, 0 };
        int[] distances = new int[2];
        long[] labels = new long[2];

        index.Search(1, query, k: 2, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);

        Assert.Equal(1, labels[1]);
        Assert.Equal(16, distances[1]);
    }

    [Fact]
    public void Constructor_InvalidDimensions_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new IndexBinaryFlat(10));
    }
}
