using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Extensions;

public class IndexExtensionsTests
{
    private const int Dimensions = 4;
    private const int Count = 8;

    [Fact]
    public void AddSpan_InfersCountFromSpanLength()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 100, count: Count, dimensions: Dimensions);

        index.Add(vectors);

        Assert.Equal(Count, index.TotalCount);
    }

    [Fact]
    public void AddSpan_LengthNotMultipleOfDimensions_Throws()
    {
        using var index = new IndexFlatL2(Dimensions);

        Assert.Throws<ArgumentException>(() => index.Add(new float[Dimensions + 1]));
        Assert.Throws<ArgumentException>(() => index.Add(ReadOnlySpan<float>.Empty));
    }

    [Fact]
    public void AddMemoryList_FlattensChunksInOrder()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 101, count: Count, dimensions: Dimensions);

        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(vectors, i * Dimensions, Dimensions));
        }

        index.Add(chunks);

        Assert.Equal(Count, index.TotalCount);

        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];

        for (var i = 0; i < Count; i++)
        {
            index.Search(1, vectors.AsSpan(i * Dimensions, Dimensions), 1, distances, labels);

            Assert.Equal(i, labels[0]);
        }
    }

    [Fact]
    public void AddMemoryList_WrongChunkLength_Throws()
    {
        using var index = new IndexFlatL2(Dimensions);
        var chunks = new List<ReadOnlyMemory<float>>
        {
            new float[Dimensions],
            new float[Dimensions - 1],
        };

        Assert.Throws<ArgumentException>(() => index.Add(chunks));
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMemoryList_Empty_IsNoOp()
    {
        using var index = new IndexFlatL2(Dimensions);

        index.Add(new List<ReadOnlyMemory<float>>());

        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMappedSingle_AssignsTheGivenId()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));
        var vector = Vectors.Random(seed: 102, count: 1, dimensions: Dimensions);

        index.Add(vector, xid: 77);

        Assert.Equal(1, index.TotalCount);

        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        index.Search(1, vector, 1, distances, labels);

        Assert.Equal(77, labels[0]);
    }

    [Fact]
    public void AddMappedList_AssignsIdsInOrder()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));
        var vectors = Vectors.Random(seed: 103, count: Count, dimensions: Dimensions);

        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(vectors, i * Dimensions, Dimensions));
        }

        long[] ids = [10, 20, 30, 40, 50, 60, 70, 80];

        index.Add(chunks, ids);

        Assert.Equal(Count, index.TotalCount);

        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];

        for (var i = 0; i < Count; i++)
        {
            index.Search(1, vectors.AsSpan(i * Dimensions, Dimensions), 1, distances, labels);

            Assert.Equal(ids[i], labels[0]);
        }
    }

    [Fact]
    public void AddMappedList_IdCountMustMatchVectorCount()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));

        var chunks = new List<ReadOnlyMemory<float>>
        {
            new float[Dimensions],
            new float[Dimensions],
        };

        Assert.Throws<ArgumentException>(() => index.Add(chunks, [1]));
        Assert.Throws<ArgumentException>(() => index.Add(chunks, [1, 2, 3]));
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMappedList_VectorCountNeedNotRelateToDimensions()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));
        var vectors = Vectors.Random(seed: 117, count: 3, dimensions: Dimensions);

        var chunks = new List<ReadOnlyMemory<float>>
        {
            new(vectors, 0, Dimensions),
            new(vectors, Dimensions, Dimensions),
            new(vectors, 2 * Dimensions, Dimensions),
        };

        index.Add(chunks, [5, 6, 7]);

        Assert.Equal(3, index.TotalCount);
    }

    [Fact]
    public void SearchPooled_MatchesLowLevelSearch()
    {
        const int k = 3;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 104, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        var expectedDistances = new float[Count * k];
        var expectedLabels = new long[Count * k];
        index.Search(Count, vectors, k, expectedDistances, expectedLabels);

        using var pooled = index.SearchPooled(vectors, k);

        Assert.Equal(Count, pooled.Count);
        Assert.Equal(k, pooled.K);

        for (var q = 0; q < Count; q++)
        {
            for (var rank = 0; rank < k; rank++)
            {
                var hit = pooled[q, rank];

                Assert.Equal(expectedLabels[(q * k) + rank], hit.Label);
                FloatAssert.Equal(expectedDistances[(q * k) + rank], hit.Distance);
            }
        }
    }

    [Fact]
    public void SearchPooled_OutOfRangeIndexers_Throw()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 105, count: 2, dimensions: Dimensions);
        index.Add(2, vectors);

        using var pooled = index.SearchPooled(vectors, k: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => pooled[2, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => pooled[0, 2]);
    }

    [Fact]
    public void SearchSpan_EnumeratesRanksInOrder()
    {
        const int k = 4;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 106, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        var expectedDistances = new float[k];
        var expectedLabels = new long[k];
        index.Search(1, vectors.AsSpan(0, Dimensions), k, expectedDistances, expectedLabels);

        using var result = index.Search(vectors.AsSpan(0, Dimensions), k);

        Assert.Equal(k, result.Length);
        Assert.Equal(expectedLabels, result.Labels.ToArray());
        FloatAssert.Equal(expectedDistances, result.Distances);

        var rank = 0;
        foreach (var hit in result)
        {
            Assert.Equal(expectedLabels[rank], hit.Label);
            FloatAssert.Equal(expectedDistances[rank], hit.Distance);
            rank++;
        }

        Assert.Equal(k, rank);
    }

    [Fact]
    public void SearchSpan_RentedBufferIsSlicedToK()
    {
        const int k = 3;

        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 107, count: Count, dimensions: Dimensions));

        using var result = index.Search(Vectors.Random(seed: 108, count: 1, dimensions: Dimensions), k);

        Assert.Equal(k, result.Labels.Length);
        Assert.Equal(k, result.Distances.Length);

        var threw = false;
        try
        {
            _ = result[k];
        }
        catch (ArgumentOutOfRangeException)
        {
            threw = true;
        }

        Assert.True(threw, $"Indexing rank {k} of a {k}-result span should throw.");
    }

    [Fact]
    public void SearchView_WritesIntoCallerBuffers()
    {
        const int k = 2;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 109, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        Span<float> distances = stackalloc float[k];
        Span<long> labels = stackalloc long[k];

        var view = index.Search(vectors.AsSpan(0, Dimensions), k, distances, labels);

        Assert.Equal(k, view.Length);
        Assert.Equal(0, view[0].Label);

        Assert.Equal(labels[0], view.Labels[0]);
        Assert.Equal(distances[1], view.Distances[1]);
    }

    [Fact]
    public void SearchView_UndersizedBuffers_Throw()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 110, count: Count, dimensions: Dimensions));

        var query = new float[Dimensions];

        Assert.Throws<ArgumentException>(() => index.Search(query, 3, new float[2], new long[3]));
        Assert.Throws<ArgumentException>(() => index.Search(query, 3, new float[3], new long[2]));
        Assert.Throws<ArgumentException>(() => index.Search(new float[Dimensions + 1], 3, new float[3], new long[3]));
    }

    [Fact]
    public void AssignSpan_MatchesLowLevelAssign()
    {
        const int k = 2;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 111, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        var expected = new long[Count * k];
        index.Assign(Count, vectors, k, expected);

        using var assigned = index.Assign(vectors, k);

        Assert.Equal(Count, assigned.QueriesCount);
        Assert.Equal(k, assigned.K);
        Assert.Equal(expected, assigned.Labels.ToArray());
    }

    [Fact]
    public void AssignView_WritesIntoCallerBuffer()
    {
        const int k = 2;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 112, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        Span<long> labels = stackalloc long[k];
        index.Assign(vectors.AsSpan(0, Dimensions), k, labels);

        Assert.Equal(0, labels[0]);
        Assert.InRange(labels[1], 0L, Count - 1L);
    }

    [Fact]
    public void SearchWithParams_MemoryOverload_MatchesLowLevelSearch()
    {
        const int k = 3;

        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 113, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(vectors, i * Dimensions, Dimensions));
        }

        var expectedDistances = new float[Count * k];
        var expectedLabels = new long[Count * k];
        index.Search(Count, vectors, k, expectedDistances, expectedLabels);

        using var parameters = new SearchParameters();
        var results = index.SearchWithParams(chunks, k, parameters);

        Assert.Equal(Count, results.QueryCount);
        Assert.Equal(k, results.K);

        for (var q = 0; q < Count; q++)
        {
            for (var rank = 0; rank < k; rank++)
            {
                var hit = results[q, rank];

                Assert.Equal(expectedLabels[(q * k) + rank], hit.Label);
                FloatAssert.Equal(expectedDistances[(q * k) + rank], hit.Distance);
            }
        }
    }

    [Fact]
    public void SearchWithParams_WrongQueryLength_Throws()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 114, count: Count, dimensions: Dimensions));

        using var parameters = new SearchParameters();

        var badFirst = new List<ReadOnlyMemory<float>>
        {
            new float[Dimensions - 1],
            new float[Dimensions],
        };
        Assert.Throws<ArgumentException>(() => index.SearchWithParams(badFirst, 2, parameters));

        var badSecond = new List<ReadOnlyMemory<float>>
        {
            new float[Dimensions],
            new float[Dimensions - 1],
        };
        Assert.Throws<ArgumentException>(() => index.SearchWithParams(badSecond, 2, parameters));
    }

    [Fact]
    public void RemoveIds_DropsTheSelectedVectors()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 115, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        var removed = index.RemoveIds([1L, 3L]);

        Assert.Equal(2, removed);
        Assert.Equal(Count - 2, index.TotalCount);
    }

    [Fact]
    public async Task TrainAsync_MemoryList_TrainsTheIndex()
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new Faiss.Cpu.Indexes.IVF.IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, nlist: 2);
        Assert.False(index.IsTrained);

        var train = Vectors.Random(seed: 116, count: 64, dimensions: Dimensions);
        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < 64; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(train, i * Dimensions, Dimensions));
        }

        await index.TrainAsync(chunks);

        Assert.True(index.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_WrongChunkLength_Throws()
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new Faiss.Cpu.Indexes.IVF.IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, nlist: 2);

        var chunks = new List<ReadOnlyMemory<float>> { new float[Dimensions + 1] };

        await Assert.ThrowsAsync<ArgumentException>(() => index.TrainAsync(chunks));
        Assert.False(index.IsTrained);
    }

    [Fact]
    public void AddSpan_InvalidLength_NamesVectors()
    {
        using var index = new IndexFlatL2(Dimensions);

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.Add(ReadOnlySpan<float>.Empty)).ParamName);
        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.Add(new float[Dimensions + 1])).ParamName);
    }

    [Fact]
    public void AddMemoryList_BadChunk_AddsNothing()
    {
        using var index = new IndexFlatL2(Dimensions);
        var chunks = new List<ReadOnlyMemory<float>> { new float[Dimensions], new float[Dimensions - 1] };

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.Add(chunks)).ParamName);
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMappedSingle_MoreThanOneVector_Throws()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));

        Assert.Equal("xids", Assert.Throws<ArgumentException>(() => index.Add(new float[2 * Dimensions], 7L)).ParamName);
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void AddMappedList_IdsWithoutVectors_Throws()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));

        Assert.Equal("xids", Assert.Throws<ArgumentException>(() => index.Add(new List<ReadOnlyMemory<float>>(), new List<long> { 1 })).ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void PooledOverloads_NonPositiveK_NameK(int k)
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 115, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        Assert.Equal("k", Assert.Throws<ArgumentOutOfRangeException>(() => { using var r = index.Search(vectors.AsSpan(0, Dimensions), k); }).ParamName);
        Assert.Equal("k", Assert.Throws<ArgumentOutOfRangeException>(() => { using var r = index.SearchPooled(vectors, k); }).ParamName);
        Assert.Equal("k", Assert.Throws<ArgumentOutOfRangeException>(() => { using var r = index.Assign(vectors, k); }).ParamName);
    }

    [Fact]
    public void QueryOverloads_WrongQueryLength_NameTheQuery()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 116, count: Count, dimensions: Dimensions));
        var ragged = new float[Dimensions + 1];

        Assert.Equal("queryVectors", Assert.Throws<ArgumentException>(() => { using var r = index.Search(ragged, 2); }).ParamName);
        Assert.Equal("queryVectors", Assert.Throws<ArgumentException>(() => { using var r = index.SearchPooled(ragged, 2); }).ParamName);
        Assert.Equal("queryVectors", Assert.Throws<ArgumentException>(() => { using var r = index.Assign(ragged, 2); }).ParamName);
        Assert.Equal("queryVector", Assert.Throws<ArgumentException>(() => index.Search(ragged, 2, new float[2], new long[2])).ParamName);
        Assert.Equal("queryVector", Assert.Throws<ArgumentException>(() => index.Assign(ragged, 2, new long[2])).ParamName);
    }

    [Fact]
    public void SearchWithParams_SingleMemory_IsOneQuery()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 117, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);
        using var parameters = new SearchParameters();

        var results = index.SearchWithParams(new ReadOnlyMemory<float>(vectors, 3 * Dimensions, Dimensions), 2, parameters);

        Assert.Equal(1, results.QueryCount);
        Assert.Equal(3, results[0, 0].Label);

        Assert.Equal("queryVectors", Assert.Throws<ArgumentException>(() => index.SearchWithParams(new ReadOnlyMemory<float>(vectors, 0, 2 * Dimensions), 2, parameters)).ParamName);
    }

    [Fact]
    public void RangeSearchSingle_MatchesBatchRangeSearch_AndRequiresOneVector()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 118, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        using var single = new RangeSearchResult(1);
        index.RangeSearch(vectors.AsSpan(0, Dimensions), 2f, single);
        using var batch = new RangeSearchResult(1);
        index.RangeSearch(1, vectors.AsSpan(0, Dimensions), 2f, batch);

        Assert.Equal(batch.GetQueryResult(0).Labels.ToArray(), single.GetQueryResult(0).Labels.ToArray());
        Assert.Contains(0L, single.GetQueryResult(0).Labels.ToArray());

        using var ignored = new RangeSearchResult(1);
        Assert.Equal("queryVector", Assert.Throws<ArgumentException>(() => index.RangeSearch(new float[Dimensions + 1], 2f, ignored)).ParamName);
    }

    [Fact]
    public void SetParameter_ChangesTheIndex_AndRejectsUnknownNames()
    {
        using var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 4, ownQuantizer: true);

        index.SetParameter("nprobe", 3);
        Assert.Equal(3, index.Nprobe);

        Assert.Throws<FaissNativeException>(() => index.SetParameter("no_such_parameter", 1));
        using var flat = new IndexFlatL2(Dimensions);
        Assert.Throws<FaissNativeException>(() => flat.SetParameter("nprobe", 2));
    }

    [Fact]
    public void RemoveIdsSpan_DuplicatesCountOnce_EmptyRemovesNothing()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 119, count: Count, dimensions: Dimensions));

        Assert.Equal(0, index.RemoveIds(ReadOnlySpan<long>.Empty));
        Assert.Equal(1, index.RemoveIds([2L, 2L]));
        Assert.Equal(Count - 1, index.TotalCount);
    }

    [Fact]
    public void RemainingValidation_NamesTheArgument()
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 120, count: Count, dimensions: Dimensions));

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => map.Add(new float[Dimensions + 1], 7L)).ParamName);
        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => map.Add(new List<ReadOnlyMemory<float>> { new float[Dimensions - 1] }, new List<long> { 1 })).ParamName);
        Assert.Equal("labels", Assert.Throws<ArgumentException>(() => index.Assign(new float[Dimensions], 3, new long[2])).ParamName);
        Assert.Equal("distances", Assert.Throws<ArgumentException>(() => index.Search(new float[Dimensions], 3, new float[2], new long[3])).ParamName);

        map.Add(new List<ReadOnlyMemory<float>>(), new List<long>());
        Assert.Equal(0, map.TotalCount);
    }
}
