using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class BinaryParamsSearchContractTests
{
    private const int Count = 8;

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchBinary), MemberType = typeof(IndexCatalog))]
    public void NullSelector_MatchesPlainSearch(BinaryIndexCase c)
    {
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var paramsIndex = Params(index);

        var plainDistances = new int[Count * k];
        var plainLabels = new long[Count * k];
        index.Search(Count, vectors, k, plainDistances, plainLabels);

        using var parameters = c.CreateParameters(null);
        var paramDistances = new int[Count * k];
        var paramLabels = new long[Count * k];
        paramsIndex.SearchWithParams(Count, vectors, k, parameters, paramDistances, paramLabels);

        Assert.Equal(plainLabels, paramLabels);
        Assert.Equal(plainDistances, paramDistances);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SelectorBinary), MemberType = typeof(IndexCatalog))]
    public void Selector_MatchesBruteForceOverAllowedIds(BinaryIndexCase c)
    {
        const int k = 2;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        long[] allowed = [1, 4, 6];
        using var selector = IDSelector.Batch(allowed);
        using var parameters = c.CreateParameters(selector);

        var distances = new int[Count * k];
        var labels = new long[Count * k];
        Params(index).SearchWithParams(Count, vectors, k, parameters, distances, labels);

        for (var q = 0; q < Count; q++)
        {
            var expected = BruteForce(c, vectors, q, k, allowed);
            Assert.Equal(expected, labels.Skip(q * k).Take(k).ToArray());
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SelectorBinary), MemberType = typeof(IndexCatalog))]
    public void Selector_FewerMatchesThanK_PadsWithSentinels(BinaryIndexCase c)
    {
        const int k = 4;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        using var selector = IDSelector.Range(3, 5);
        using var parameters = c.CreateParameters(selector);

        var distances = new int[k];
        var labels = new long[k];
        Params(index).SearchWithParams(1, vectors.AsSpan(0, c.CodeSize), k, parameters, distances, labels);

        Assert.All(labels.Take(2), label => Assert.InRange(label, 3L, 4L));
        Assert.Equal(-1, labels[2]);
        Assert.Equal(-1, labels[3]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchBinary), MemberType = typeof(IndexCatalog))]
    public void InvalidArguments_Throw(BinaryIndexCase c)
    {
        const int queries = 2;
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var paramsIndex = Params(index);

        using var parameters = c.CreateParameters(null);
        var distances = new int[queries * k];
        var labels = new long[queries * k];

        Assert.Throws<ArgumentNullException>(() => paramsIndex.SearchWithParams(queries, vectors, k, null!, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => paramsIndex.SearchWithParams(-1, vectors, k, parameters, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => paramsIndex.SearchWithParams(queries, vectors, 0, parameters, distances, labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors.AsSpan(0, c.CodeSize), k, parameters, distances, labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors, k, parameters, distances.AsSpan(0, queries * k - 1), labels));
        Assert.Throws<ArgumentException>(() => paramsIndex.SearchWithParams(queries, vectors, k, parameters, distances, labels.AsSpan(0, queries * k - 1)));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ParamsSearchBinary), MemberType = typeof(IndexCatalog))]
    public void DisposedParameters_Throw(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        var parameters = c.CreateParameters(null);
        parameters.Dispose();

        var distances = new int[1];
        var labels = new long[1];

        Assert.Throws<ObjectDisposedException>(
            () => Params(index).SearchWithParams(1, vectors.AsSpan(0, c.CodeSize), 1, parameters, distances, labels)
        );
    }

    private static byte[] Populate(BinaryIndex index, BinaryIndexCase c)
    {
        var vectors = Vectors.RandomBinary(seed: 33, count: Count, bitDimensions: c.Dimensions);
        Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index).Add(Count, vectors);
        return vectors;
    }

    private static IParamsBinarySearchIndex Params(BinaryIndex index) =>
        Assert.IsAssignableFrom<IParamsBinarySearchIndex>(index);

    /// <summary>True top-k by Hamming distance over <paramref name="allowed"/> in the same way faiss ties broken by ascending id.</summary>
    private static long[] BruteForce(BinaryIndexCase c, byte[] vectors, int queryIndex, int k, long[] allowed)
    {
        var query = vectors.AsSpan(queryIndex * c.CodeSize, c.CodeSize);
        var scored = new List<(long Id, int Distance)>();

        foreach (var id in allowed)
        {
            scored.Add((id, Oracles.Hamming(query, vectors.AsSpan((int)id * c.CodeSize, c.CodeSize))));
        }

        return scored.OrderBy(s => s.Distance).ThenBy(s => s.Id).Take(k).Select(s => s.Id).ToArray();
    }
}
