using Faiss.Cpu.Indexes.Binary;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Binary;

public class IndexBinaryFlatTests
{
    private const int Dimensions = 16;
    private const int CodeSize = Dimensions / 8;

    [Fact]
    public void Constructor_DimensionsNotMultipleOf8_Throws()
    {
        Assert.Throws<ArgumentException>(() => new IndexBinaryFlat(10));
    }

    [Fact]
    public void Search_K2_MatchesHammingOracle()
    {
        using var index = new IndexBinaryFlat(Dimensions);
        byte[] xb =
        [
            255, 0,
            0, 255,
        ];
        index.Add(2, xb);

        byte[] query = [255, 0];
        Span<int> distances = stackalloc int[2];
        Span<long> labels = stackalloc long[2];
        index.Search(1, query, k: 2, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);
        Assert.Equal(Oracles.Hamming(query, xb.AsSpan(0, CodeSize)), distances[0]);
        Assert.Equal(Oracles.Hamming(query, xb.AsSpan(CodeSize, CodeSize)), distances[1]);
    }
}
