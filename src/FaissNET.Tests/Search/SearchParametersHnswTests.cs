using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Search;

public class SearchParametersHnswTests
{
    private const int Dimensions = 8;
    private const int Count = 32;

    [Fact]
    public void Properties_SurviveConstructionAndAssignment()
    {
        using var parameters = new SearchParametersHNSW(efSearch: 48);
        Assert.Equal(48, parameters.EfSearch);

        parameters.EfSearch = 99;
        Assert.Equal(99, parameters.EfSearch);

        var relative = parameters.CheckRelativeDistance;
        parameters.CheckRelativeDistance = !relative;
        Assert.Equal(!relative, parameters.CheckRelativeDistance);
        parameters.CheckRelativeDistance = relative;
        Assert.Equal(relative, parameters.CheckRelativeDistance);

        var bounded = parameters.BoundedQueue;
        parameters.BoundedQueue = !bounded;
        Assert.Equal(!bounded, parameters.BoundedQueue);
    }

    [Fact]
    public void WideFrontier_ReturnsTrueNearestNeighbours()
    {
        const int k = 3;

        using var index = new IndexHNSWFlat(Dimensions, m: 32);
        var vectors = Populate(index);

        using var parameters = new SearchParametersHNSW(efSearch: Count);
        var distances = new float[k];
        var labels = new long[k];

        index.SearchWithParams(1, vectors.AsSpan(0, Dimensions), k, parameters, distances, labels);

        Assert.Equal(BruteForce(vectors, queryIndex: 0, k, allowed: null), labels);
    }

    [Fact]
    public void Selector_RestrictsResultsToMembers()
    {
        const int k = 2;

        using var index = new IndexHNSWFlat(Dimensions, m: 32);
        var vectors = Populate(index);

        long[] allowed = [4, 9, 17];
        using var selector = IDSelector.Batch(allowed);
        using var parameters = new SearchParametersHNSW(efSearch: Count, selector);

        var distances = new float[k];
        var labels = new long[k];
        index.SearchWithParams(1, vectors.AsSpan(0, Dimensions), k, parameters, distances, labels);

        Assert.Equal(BruteForce(vectors, queryIndex: 0, k, allowed), labels);
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var parameters = new SearchParametersHNSW(efSearch: 16);
        parameters.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = parameters.EfSearch);
    }

    private static float[] Populate(IndexHNSWFlat index)
    {
        var vectors = Vectors.Random(seed: 30, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);
        return vectors;
    }

    private static long[] BruteForce(float[] vectors, int queryIndex, int k, long[]? allowed)
    {
        var query = vectors.AsSpan(queryIndex * Dimensions, Dimensions);
        var scored = new List<(long Id, float Score)>();

        for (long i = 0; i < Count; i++)
        {
            if (allowed is not null && !allowed.Contains(i))
            {
                continue;
            }

            scored.Add((i, Oracles.L2Sqr(query, vectors.AsSpan((int)i * Dimensions, Dimensions))));
        }

        return scored.OrderBy(s => s.Score).Take(k).Select(s => s.Id).ToArray();
    }
}
