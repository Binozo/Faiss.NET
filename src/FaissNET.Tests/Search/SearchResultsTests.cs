using System.Buffers;
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Search;

public class SearchResultsTests
{
    private const int Dimensions = 4;
    private const int Stored = 6;
    private const int Queries = 3;
    private const int K = 4;

    private static readonly float[] Data = Vectors.Random(seed: 951, count: Stored, dimensions: Dimensions);
    private static readonly float[] QueryData = Vectors.Random(seed: 952, count: Queries, dimensions: Dimensions);

    [Fact]
    public void GetQueryResults_SlicesExactlyThatQuerysRanks()
    {
        using var index = Populated();
        var (expectedDistances, expectedLabels) = LowLevel(index, K);

        var results = Search(index, K);

        for (var q = 0; q < Queries; q++)
        {
            var row = results.GetQueryResults(q);

            Assert.Equal(expectedLabels.AsSpan(q * K, K).ToArray(), row.Labels.ToArray());
            FloatAssert.Equal(expectedDistances.AsSpan(q * K, K), row.Distances);
        }
    }

    [Fact]
    public void ToArrays_HasOneRowPerQueryInRankOrder()
    {
        using var index = Populated();
        var (expectedDistances, expectedLabels) = LowLevel(index, K);

        var rows = Search(index, K).ToArrays();

        Assert.Equal(Queries, rows.Length);
        for (var q = 0; q < Queries; q++)
        {
            Assert.Equal(K, rows[q].Length);
            for (var rank = 0; rank < K; rank++)
            {
                Assert.Equal(expectedLabels[(q * K) + rank], rows[q][rank].Label);
                FloatAssert.Equal(expectedDistances[(q * K) + rank], rows[q][rank].Distance);
            }
        }
    }

    [Fact]
    public void DistancesAndLabels_ExposeTheFlatQueryMajorBuffers()
    {
        using var index = Populated();
        var (expectedDistances, expectedLabels) = LowLevel(index, K);

        var results = Search(index, K);

        Assert.Equal(expectedLabels, results.Labels.ToArray());
        FloatAssert.Equal(expectedDistances, results.Distances.Span);
    }

    [Fact]
    public void OutOfRangeQueryOrRank_Throws()
    {
        using var index = Populated();
        var results = Search(index, K);

        Assert.Throws<ArgumentOutOfRangeException>(() => results[-1, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => results[Queries, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => results[0, -1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => results[0, K]);
        Assert.Throws<ArgumentOutOfRangeException>(() => { results.GetQueryResults(-1); });
        Assert.Throws<ArgumentOutOfRangeException>(() => { results.GetQueryResults(Queries); });
    }

    [Fact]
    public void KBeyondStoredVectors_PadsWithMinusOneLabels()
    {
        const int k = Stored + 2;
        using var index = Populated();

        var results = Search(index, k);

        for (var q = 0; q < Queries; q++)
        {
            var row = results.GetQueryResults(q);
            Assert.All(row.Labels[..Stored].ToArray(), label => Assert.InRange(label, 0L, Stored - 1L));
            Assert.Equal([-1L, -1L], row.Labels[Stored..].ToArray());
            Assert.Equal([float.MaxValue, float.MaxValue], row.Distances[Stored..].ToArray());
        }
    }

    [Fact]
    public void NoQueries_YieldsEmptyResults()
    {
        using var index = Populated();
        using var parameters = new SearchParameters();

        var results = index.SearchWithParams(new List<ReadOnlyMemory<float>>(), K, parameters);

        Assert.Equal(0, results.QueryCount);
        Assert.Empty(results.ToArrays());
        Assert.Throws<ArgumentOutOfRangeException>(() => { results.GetQueryResults(0); });
    }

    [Fact]
    public void SearchResultView_EnumeratesEveryRankInOrder()
    {
        using var index = Populated();
        var (expectedDistances, expectedLabels) = LowLevel(index, K);

        Span<float> distances = stackalloc float[K];
        Span<long> labels = stackalloc long[K];
        var view = index.Search(QueryData.AsSpan(0, Dimensions), K, distances, labels);

        var rank = 0;
        foreach (var hit in view)
        {
            Assert.Equal(expectedLabels[rank], hit.Label);
            FloatAssert.Equal(expectedDistances[rank], hit.Distance);
            rank++;
        }

        Assert.Equal(K, rank);
    }

    [Fact]
    public void SearchResultView_OutOfRangeRank_Throws()
    {
        using var index = Populated();

        Span<float> distances = stackalloc float[K];
        Span<long> labels = stackalloc long[K];
        var view = index.Search(QueryData.AsSpan(0, Dimensions), K, distances, labels);

        Assert.True(Throws(view, K), $"Rank {K} of a {K}-result view must throw.");
        Assert.True(Throws(view, -1), "Rank -1 must throw.");

        static bool Throws(SearchResultView view, int rank)
        {
            try
            {
                _ = view[rank];
                return false;
            }
            catch (ArgumentOutOfRangeException)
            {
                return true;
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void SearchWithParams_NonPositiveK_Throws(int k)
    {
        using var index = Populated();

        Assert.Throws<ArgumentOutOfRangeException>(() => Search(index, k));
    }

    [Fact]
    public void Pooled_DisposedTwice_ReturnsItsArraysOnce()
    {
        using var index = Populated();
        var results = index.SearchPooled(QueryData, K);

        results.Dispose();
        results.Dispose();

        Assert.False(PoolHandsOutTheSameArrayTwice(Queries * K), "The labels array was returned to the pool twice.");
    }

    [Fact]
    public void Pooled_AfterDispose_Throws()
    {
        using var index = Populated();
        var results = index.SearchPooled(QueryData, K);
        results.Dispose();

        Assert.Throws<ObjectDisposedException>(() => results[0, 0]);
    }

    [Fact]
    public void Pooled_Default_DisposesWithoutError()
    {
        default(PooledSearchResults).Dispose();
    }

    [Fact]
    public void Span_DisposedTwice_ReturnsItsArraysOnce()
    {
        using var index = Populated();
        var result = index.Search(QueryData.AsSpan(0, Dimensions), K);

        result.Dispose();
        result.Dispose();

        Assert.False(PoolHandsOutTheSameArrayTwice(K), "The labels array was returned to the pool twice.");
    }

    [Fact]
    public void Span_AfterDispose_Throws()
    {
        using var index = Populated();
        var result = index.Search(QueryData.AsSpan(0, Dimensions), K);
        result.Dispose();

        Assert.True(Throws(result, r => _ = r[0]), "Indexer after Dispose must throw.");
        Assert.True(Throws(result, r => _ = r.Labels), "Labels after Dispose must throw.");
        Assert.True(Throws(result, r => _ = r.Distances), "Distances after Dispose must throw.");

        static bool Throws(SearchResultSpan span, SpanAccess access)
        {
            try
            {
                access(span);
                return false;
            }
            catch (ObjectDisposedException)
            {
                return true;
            }
        }
    }

    [Fact]
    public void Assign_DisposedTwice_ReturnsItsArrayOnce()
    {
        using var index = Populated();
        var result = index.Assign(QueryData, K);

        result.Dispose();
        result.Dispose();

        Assert.False(PoolHandsOutTheSameArrayTwice(Queries * K), "The labels array was returned to the pool twice.");
    }

    [Fact]
    public void Assign_AfterDispose_Throws()
    {
        using var index = Populated();
        var result = index.Assign(QueryData, K);
        result.Dispose();

        var threw = false;
        try
        {
            _ = result.Labels;
        }
        catch (ObjectDisposedException)
        {
            threw = true;
        }

        Assert.True(threw, "Labels after Dispose must throw.");
    }

    private delegate void SpanAccess(SearchResultSpan span);

    private static bool PoolHandsOutTheSameArrayTwice(int length)
    {
        var first = ArrayPool<long>.Shared.Rent(length);
        var second = ArrayPool<long>.Shared.Rent(length);
        var same = ReferenceEquals(first, second);

        ArrayPool<long>.Shared.Return(first);
        if (!same)
        {
            ArrayPool<long>.Shared.Return(second);
        }

        return same;
    }

    private static IndexFlatL2 Populated()
    {
        var index = new IndexFlatL2(Dimensions);
        index.Add(Stored, Data);
        return index;
    }

    private static SearchResults Search(IndexFlatL2 index, int k)
    {
        var queries = new List<ReadOnlyMemory<float>>();
        for (var q = 0; q < Queries; q++)
        {
            queries.Add(new ReadOnlyMemory<float>(QueryData, q * Dimensions, Dimensions));
        }

        using var parameters = new SearchParameters();
        return index.SearchWithParams(queries, k, parameters);
    }

    private static (float[] Distances, long[] Labels) LowLevel(IndexFlatL2 index, int k)
    {
        var distances = new float[Queries * k];
        var labels = new long[Queries * k];
        index.Search(Queries, QueryData, k, distances, labels);

        return (distances, labels);
    }
}
