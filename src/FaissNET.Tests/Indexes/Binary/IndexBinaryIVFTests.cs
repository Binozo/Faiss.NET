using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Binary;

public class IndexBinaryIVFTests
{
    private const int Dimensions = 16;
    private const int CodeSize = Dimensions / 8;
    private const int Nlist = 2;

    [Fact]
    public void Constructor_SetsNlistAndDefaultNprobe()
    {
        using var index = new IndexBinaryIVF(Dimensions, Nlist, hnswM: null);

        Assert.Equal(Nlist, index.Nlist);
        Assert.Equal(1, index.NProbe);

        index.NProbe = Nlist;
        Assert.Equal(Nlist, index.NProbe);
    }

    [Fact]
    public async Task SearchWithParams_BaseParameters_Rejected()
    {
        using var index = await CreateAsync();
        using var parameters = new SearchParameters();

        var distances = new int[1];
        var labels = new long[1];

        var ex = Assert.Throws<FaissNativeException>(
            () => index.SearchWithParams(1, new byte[CodeSize], 1, parameters, distances, labels)
        );

        Assert.Contains("incorrect type", ex.Message);
    }

    [Fact]
    public async Task SearchWithParams_Selector_Rejected()
    {
        using var index = await CreateAsync();
        using var selector = IDSelector.Batch(0, 1);
        using var parameters = new SearchParametersIVF(nprobe: Nlist, maxCodes: 0, selector);

        var distances = new int[1];
        var labels = new long[1];

        // No selector support
        var ex = Assert.Throws<FaissNativeException>(
            () => index.SearchWithParams(1, new byte[CodeSize], 1, parameters, distances, labels)
        );

        Assert.Contains("IDSelector is not supported", ex.Message);
    }

    [Fact]
    public async Task MaxCodes_CapsTheScannedCodes()
    {
        using var index = await CreateAsync();
        var query = Vectors.RandomBinary(seed: 42, count: 1, bitDimensions: Dimensions);
        var distances = new int[8];
        var labels = new long[8];

        index.Search(1, query, 8, distances, labels);
        Assert.DoesNotContain(-1L, labels);

        index.MaxCodes = 3;
        index.Search(1, query, 8, distances, labels);
        Assert.Equal(3, labels.Count(label => label >= 0));
    }

    [Fact]
    public async Task ImbalanceFactor_EmptyIsNaN_SingleListIsNlist()
    {
        using var empty = new IndexBinaryIVF(Dimensions, Nlist, hnswM: null);
        await empty.TrainAsync(64, Vectors.RandomBinary(seed: 40, count: 64, bitDimensions: Dimensions));
        Assert.True(double.IsNaN(empty.ImbalanceFactor), "An empty index has no distribution (0/0).");

        empty.Add(4, new byte[4 * CodeSize]);
        Assert.Equal(Nlist, empty.ImbalanceFactor, precision: 9);
    }

    // Note that counting search with heap off (UseHeap = false) treats MaxCodes = 0 as "scan nothing" instead of "unlimited"
    // while the heap path (faiss IndexBinaryIVF_impl.h:195-260 vs IndexBinaryIVF.cpp:369-372) silently only returns -1.

    [Fact]
    public async Task CountingSearch_WithoutCodeBudget_Throws()
    {
        using var index = await CreateAsync();
        index.UseHeap = false;
        var query = new byte[CodeSize];

        Assert.Throws<InvalidOperationException>(() => index.Search(1, query, 2, new int[2], new long[2]));
        Assert.Throws<InvalidOperationException>(() => index.Assign(1, query, 2, new long[2]));
        using var parameters = new SearchParametersIVF(nprobe: Nlist, maxCodes: 0);
        Assert.Throws<InvalidOperationException>(() => index.SearchWithParams(1, query, 2, parameters, new int[2], new long[2]));
    }

    [Fact]
    public async Task CountingSearch_PerCallCodeBudget_OverridesTheIndexDefault()
    {
        using var index = await CreateAsync();
        index.UseHeap = false;
        using var parameters = new SearchParametersIVF(nprobe: Nlist, maxCodes: int.MaxValue);
        var labels = new long[2];

        index.SearchWithParams(1, Vectors.RandomBinary(seed: 46, count: 1, bitDimensions: Dimensions), 2, parameters, new int[2], labels);

        Assert.DoesNotContain(-1L, labels);
    }

    [Fact]
    public async Task CountingSearch_WithCodeBudget_MatchesHeapDistances()
    {
        using var index = await CreateAsync();
        var queries = Vectors.RandomBinary(seed: 43, count: 3, bitDimensions: Dimensions);
        var (heapDistances, _) = Search(index, queries, 3, k: 4);

        index.UseHeap = false;
        index.MaxCodes = int.MaxValue;
        var (countDistances, labels) = Search(index, queries, 3, k: 4);

        Assert.Equal(heapDistances, countDistances);
        Assert.DoesNotContain(-1L, labels);
    }

    [Fact]
    public async Task PerInvlistSearch_FewerQueriesThanLists_Throws()
    {
        using var index = await CreateAsync();
        index.PerInvlistSearch = true;

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(1, new byte[CodeSize], 2, new int[2], new long[2]));
    }

    [Fact]
    public async Task PerInvlistSearch_EnoughProbesButFewerQueriesThanLists_Throws()
    {
        using var index = new IndexBinaryIVF(Dimensions, nlist: 4, hnswM: null);
        await index.TrainAsync(64, Vectors.RandomBinary(seed: 47, count: 64, bitDimensions: Dimensions));
        index.Add(8, Vectors.RandomBinary(seed: 48, count: 8, bitDimensions: Dimensions));
        index.NProbe = 4;
        index.PerInvlistSearch = true;

        Assert.Throws<ArgumentOutOfRangeException>(() => index.Search(2, new byte[2 * CodeSize], 2, new int[4], new long[4]));
    }

    [Fact]
    public async Task PerInvlistSearch_WithCodeBudget_Throws()
    {
        using var index = await CreateAsync();
        index.PerInvlistSearch = true;
        index.MaxCodes = 3;
        var queries = Vectors.RandomBinary(seed: 44, count: 4, bitDimensions: Dimensions);

        Assert.Throws<InvalidOperationException>(() => index.Search(4, queries, 2, new int[8], new long[8]));
    }

    [Fact]
    public async Task PerInvlistSearch_BatchOfAtLeastNlistQueries_MatchesHeapSearch()
    {
        using var index = await CreateAsync();
        var queries = Vectors.RandomBinary(seed: 45, count: 4, bitDimensions: Dimensions);
        var (heapDistances, _) = Search(index, queries, 4, k: 3);

        index.PerInvlistSearch = true;
        var (perListDistances, _) = Search(index, queries, 4, k: 3);

        Assert.Equal(heapDistances, perListDistances);
    }

    private static (int[] Distances, long[] Labels) Search(IndexBinaryIVF index, byte[] queries, int count, int k)
    {
        var distances = new int[count * k];
        var labels = new long[count * k];
        index.Search(count, queries, k, distances, labels);
        return (distances, labels);
    }

    private static async Task<IndexBinaryIVF> CreateAsync()
    {
        var index = new IndexBinaryIVF(Dimensions, Nlist, hnswM: null);
        var train = Vectors.RandomBinary(seed: 40, count: 64, bitDimensions: Dimensions);
        await index.TrainAsync(64, train);

        index.NProbe = Nlist;
        index.Add(8, Vectors.RandomBinary(seed: 41, count: 8, bitDimensions: Dimensions));

        return index;
    }
}
