using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class ParamsSearchContractTests
{
    private const int Count = 8;

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchFloat), MemberType = typeof(IndexCatalog))]
    public void NullSelector_MatchesPlainSearch(IndexCase c)
    {
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var paramsIndex = Params(index);

        var plainDistances = new float[Count * k];
        var plainLabels = new long[Count * k];
        index.Search(Count, vectors, k, plainDistances, plainLabels);

        using var parameters = c.CreateParameters(null);
        var paramDistances = new float[Count * k];
        var paramLabels = new long[Count * k];
        paramsIndex.SearchWithParams(Count, vectors, k, parameters, paramDistances, paramLabels);

        Assert.Equal(plainLabels, paramLabels);
        Assert.Equal(plainDistances, paramDistances);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ApproximateSelectorFloat), MemberType = typeof(IndexCatalog))]
    public void Selector_ApproximateIndex_ReturnsOnlyMatchingIds(IndexCase c)
    {
        const int k = 2;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        using var selector = IDSelector.Batch(1, 3, 6);
        using var parameters = c.CreateParameters(selector);

        var distances = new float[Count * k];
        var labels = new long[Count * k];
        Params(index).SearchWithParams(Count, vectors, k, parameters, distances, labels);

        Assert.All(labels, label => Assert.True(selector.IsMember(label), $"Label {label} is not a member of the selector."));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SelectorFloat), MemberType = typeof(IndexCatalog))]
    public void Selector_MatchesBruteForceOracleOverAllowedIds(IndexCase c)
    {
        const int k = 2;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        long[] allowed = [1, 3, 6];
        using var selector = IDSelector.Batch(allowed);
        using var parameters = c.CreateParameters(selector);

        var distances = new float[Count * k];
        var labels = new long[Count * k];
        Params(index).SearchWithParams(Count, vectors, k, parameters, distances, labels);

        for (var q = 0; q < Count; q++)
        {
            var query = vectors.AsSpan(q * c.Dimensions, c.Dimensions);
            var expected = RankAllowed(c, query, vectors, allowed).Take(k).ToArray();

            Assert.Equal(expected, labels.Skip(q * k).Take(k).ToArray());
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SelectorFloat), MemberType = typeof(IndexCatalog))]
    public void Selector_FewerMatchesThanK_PadsWithSentinels(IndexCase c)
    {
        const int k = 4;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        using var selector = IDSelector.Range(2, 4);
        using var parameters = c.CreateParameters(selector);

        var distances = new float[k];
        var labels = new long[k];
        Params(index).SearchWithParams(1, vectors.AsSpan(0, c.Dimensions), k, parameters, distances, labels);

        Assert.All(labels.Take(2), label => Assert.InRange(label, 2L, 3L));
        Assert.Equal(-1, labels[2]);
        Assert.Equal(-1, labels[3]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SelectorFloat), MemberType = typeof(IndexCatalog))]
    public void NegatedSelector_ExcludesMatchingIds(IndexCase c)
    {
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        using var selector = IDSelector.Batch(0, 1, 2).Not();
        using var parameters = c.CreateParameters(selector);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var distances = new float[k];
        var labels = new long[k];
        Params(index).SearchWithParams(1, vectors.AsSpan(0, c.Dimensions), k, parameters, distances, labels);

        long[] allowed = [3, 4, 5, 6, 7];
        var expected = RankAllowed(c, vectors.AsSpan(0, c.Dimensions), vectors, allowed).Take(k).ToArray();

        Assert.Equal(expected, labels);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchFloat), MemberType = typeof(IndexCatalog))]
    public void InvalidArguments_Throw(IndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var paramsIndex = Params(index);

        using var parameters = c.CreateParameters(null);
        var distances = new float[queries * k];
        var labels = new long[queries * k];

        Assert.Throws<ArgumentNullException>(() => paramsIndex.SearchWithParams(queries, vectors, k, null!, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => paramsIndex.SearchWithParams(-1, vectors, k, parameters, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => paramsIndex.SearchWithParams(queries, vectors, 0, parameters, distances, labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors.AsSpan(0, c.Dimensions), k, parameters, distances, labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors, k, parameters, distances.AsSpan(0, queries * k - 1), labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors, k, parameters, distances, labels.AsSpan(0, queries * k - 1)));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchFloat), MemberType = typeof(IndexCatalog))]
    public void DisposedParameters_Throw(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        var parameters = c.CreateParameters(null);
        parameters.Dispose();

        var distances = new float[1];
        var labels = new long[1];

        Assert.Throws<ObjectDisposedException>(() => Params(index).SearchWithParams(1, vectors.AsSpan(0, c.Dimensions), 1, parameters, distances, labels));
    }

    private static float[] Populate(FloatIndex index, IndexCase c)
    {
        var vectors = Vectors.Random(seed: 11, count: Count, dimensions: c.Dimensions);
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Count, vectors);
        return vectors;
    }

    private static IParamsFloatSearchIndex Params(FloatIndex index) => Assert.IsAssignableFrom<IParamsFloatSearchIndex>(index);

    /// <summary>Allowed ids ranked exactly as faiss would rank them for the case metric.</summary>
    private static IEnumerable<long> RankAllowed(IndexCase c, ReadOnlySpan<float> query, float[] vectors, long[] allowed)
    {
        var scored = new List<(long Id, float Score)>(allowed.Length);
        foreach (var id in allowed)
        {
            var candidate = vectors.AsSpan((int)id * c.Dimensions, c.Dimensions);
            scored.Add((id, c.Metric == MetricType.InnerProduct
                ? Oracles.InnerProduct(query, candidate)
                : Oracles.L2Sqr(query, candidate)));
        }

        return c.Metric == MetricType.InnerProduct
            ? scored.OrderByDescending(s => s.Score).Select(s => s.Id)
            : scored.OrderBy(s => s.Score).Select(s => s.Id);
    }
}
