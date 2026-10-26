using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Search;

public class BinarySearchParametersTests
{
    private const int Bits = 16;
    private const int CodeSize = Bits / 8;
    private const int Count = 32;

    [Fact]
    public void BinaryHnsw_WideFrontier_FindsTheExactNearestNeighbour()
    {
        using var index = new IndexBinaryHNSW(Bits, m: 16);
        var vectors = Populate(index);

        using var parameters = new SearchParametersHNSW(efSearch: Count);

        var distances = new int[1];
        var labels = new long[1];
        index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 1, parameters, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);
    }

    [Fact]
    public void BinaryHnsw_EfSearchIsForwarded()
    {
        using var index = new IndexBinaryHNSW(Bits, m: 16);
        var vectors = Populate(index);

        using var narrow = new SearchParametersHNSW(efSearch: 1);
        using var wide = new SearchParametersHNSW(efSearch: Count);

        Assert.Equal(1, narrow.EfSearch);
        Assert.Equal(Count, wide.EfSearch);

        var narrowLabels = new long[4];
        var wideLabels = new long[4];
        index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 4, narrow, new int[4], narrowLabels);
        index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 4, wide, new int[4], wideLabels);

        Assert.Equal(0, wideLabels[0]);
        Assert.All(narrowLabels, l => Assert.InRange(l, -1L, Count - 1L));
    }

    [Fact]
    public void BinaryHnsw_BaseParameters_Rejected()
    {
        using var index = new IndexBinaryHNSW(Bits, m: 16);
        var vectors = Populate(index);

        using var parameters = new SearchParameters();

        var ex = Assert.Throws<FaissNativeException>(
            () => index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 1, parameters, new int[1], new long[1])
        );

        Assert.Contains("incorrect type", ex.Message);
    }

    [Fact]
    public void BinaryHnsw_SelectorRestrictsResults()
    {
        using var index = new IndexBinaryHNSW(Bits, m: 16);
        var vectors = Populate(index);

        long[] allowed = [5, 9, 21];
        using var selector = IDSelector.Batch(allowed);
        using var parameters = new SearchParametersHNSW(efSearch: Count, selector);

        var labels = new long[3];
        index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 3, parameters, new int[3], labels);

        Assert.All(labels, label => Assert.Contains(label, allowed));
    }

    [Fact]
    public async Task BinaryIvf_ParametersOverrideIndexNprobe()
    {
        using var index = new IndexBinaryIVF(Bits, nlist: 4, hnswM: null);
        await index.TrainAsync(256, Vectors.RandomBinary(seed: 180, count: 256, bitDimensions: Bits));

        var vectors = Vectors.RandomBinary(seed: 181, count: Count, bitDimensions: Bits);
        index.Add(Count, vectors);
        index.NProbe = 1;

        using var parameters = new SearchParametersIVF(nprobe: 4);

        var labels = new long[1];
        index.SearchWithParams(1, vectors.AsSpan(0, CodeSize), 1, parameters, new int[1], labels);

        Assert.Equal(0, labels[0]);

        Assert.Equal(1, index.NProbe);
    }

    private static byte[] Populate(IndexBinaryHNSW index)
    {
        var vectors = Vectors.RandomBinary(seed: 182, count: Count, bitDimensions: Bits);
        index.Add(Count, vectors);

        return vectors;
    }
}
