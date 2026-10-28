using Faiss.Models;
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class RemoveIdsContractTests
{
    private const int Dimensions = 4;
    private const int Count = 6;

    public sealed record SelectorCase(string Name, Func<long[], IDSelector> Create, int[] Removed)
    {
        public override string ToString() => Name;
    }

    private static readonly SelectorCase[] Selectors =
    [
        new("Batch", ids => IDSelector.Batch(ids[1], ids[3]), [1, 3]),
        new("Range", ids => IDSelector.Range(ids[1], ids[3]), [1, 2]),
    ];

    public static TheoryData<IndexCase, SelectorCase> RemovableFloatWithSelectors { get; } = Cross(IndexCatalog.RemovableFloat);

    public static TheoryData<BinaryIndexCase, SelectorCase> RemovableBinaryWithSelectors { get; } = Cross(IndexCatalog.RemovableBinary);

    [Theory]
    [MemberData(nameof(RemovableFloatWithSelectors))]
    public void RemoveIds_RemovesTheSelected_AndLabelsSurvivorsPerFamily(IndexCase c, SelectorCase s)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 129, count: Count, dimensions: c.Dimensions);
        var ids = PopulateFloat(c, index, vectors);

        using var selector = s.Create(ids);
        Assert.Equal(s.Removed.Length, Removable(index).RemoveIds(selector));
        Assert.Equal(Count - s.Removed.Length, index.TotalCount);

        var expected = ExpectedLabels(c, ids, s.Removed);
        var k = Count - s.Removed.Length;
        var distances = new float[Count * k];
        var labels = new long[Count * k];
        index.Search(Count, vectors, k, distances, labels);

        AssertSurvivorLabels(c.Name, c.Metric, expected, labels, k);
    }

    [Theory]
    [MemberData(nameof(RemovableBinaryWithSelectors))]
    public void RemoveIds_RemovesTheSelected_AndLabelsSurvivorsPerFamily_Binary(BinaryIndexCase c, SelectorCase s)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 130, count: Count, bitDimensions: c.Dimensions);
        var ids = PopulateBinary(c, index, vectors);

        using var selector = s.Create(ids);
        Assert.Equal(s.Removed.Length, ((IIDRemovableBinaryIndex)index).RemoveIds(selector));
        Assert.Equal(Count - s.Removed.Length, index.TotalCount);

        var expected = ExpectedLabels(c, ids, s.Removed);
        var k = Count - s.Removed.Length;
        var distances = new int[Count * k];
        var labels = new long[Count * k];
        index.Search(Count, vectors, k, distances, labels);

        AssertSurvivorLabels(c.Name, c.Metric, expected, labels, k);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.RemovableFloat), MemberType = typeof(IndexCatalog))]
    public void RemoveIds_MatchingNothing_RemovesNothing(IndexCase c)
    {
        using var index = c.CreateIndex();
        var ids = PopulateFloat(c, index, Vectors.Random(seed: 123, count: Count, dimensions: c.Dimensions));

        using var selector = IDSelector.Range(ids[^1] + 10, ids[^1] + 20);

        Assert.Equal(0, Removable(index).RemoveIds(selector));
        Assert.Equal(Count, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.RemovableFloat), MemberType = typeof(IndexCatalog))]
    public void RemoveIds_MatchingEverything_EmptiesTheIndex(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 124, count: Count, dimensions: c.Dimensions);
        var ids = PopulateFloat(c, index, vectors);

        using var selector = IDSelector.Range(ids[0], ids[^1] + 1);

        Assert.Equal(Count, Removable(index).RemoveIds(selector));
        Assert.Equal(0, index.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, c.Dimensions), 1, distances, labels);

        Assert.Equal(-1, labels[0]);
    }

    [Fact]
    public void Flat_RemoveIds_SelectorMatchesPositionsNotStoredValues()
    {
        using var index = new IndexFlatL2(Dimensions);
        var vectors = Vectors.Random(seed: 121, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        using var selector = IDSelector.Range(2, 5);
        var removed = index.RemoveIds(selector);

        Assert.Equal(3, removed);
        Assert.Equal(3, index.TotalCount);

        int[] survivors = [0, 1, 5];
        for (var newLabel = 0; newLabel < survivors.Length; newLabel++)
        {
            FloatAssert.Equal(vectors.AsSpan(survivors[newLabel] * Dimensions, Dimensions), index.Reconstruct(newLabel));
        }
    }

    [Fact]
    public void Flat_RemoveIds_ComposedSelector_RemovesTheIntersection()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 122, count: Count, dimensions: Dimensions));

        using var range = IDSelector.Range(0, 4);
        using var batch = IDSelector.Batch(1, 3, 5);
        using var selector = range.And(batch);

        // Intersection of [0,4) and {1,3,5} is {1,3}.
        Assert.Equal(2, index.RemoveIds(selector));
        Assert.Equal(Count - 2, index.TotalCount);
    }

    [Fact]
    public void Flat_RemoveIds_MatchingNothing_RemovesNothing()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 123, count: Count, dimensions: Dimensions));

        using var selector = IDSelector.Range(Count + 10, Count + 20);

        Assert.Equal(0, index.RemoveIds(selector));
        Assert.Equal(Count, index.TotalCount);
    }

    [Fact]
    public void IdMap_RemoveIds_OrdinalsAreNotUserIds()
    {
        using var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions));
        var vectors = Vectors.Random(seed: 126, count: 4, dimensions: Dimensions);

        index.Add(4, vectors, [100, 200, 300, 400]);

        // Ordinal-style ids match nothing, because the IDMap compares user ids.
        Assert.Equal(0, index.RemoveIds([0L, 1L, 2L, 3L]));
        Assert.Equal(4, index.TotalCount);
    }

    [Fact]
    public void RemoveIds_NullSelector_Throws()
    {
        using var index = new IndexFlatL2(Dimensions);
        index.Add(Count, Vectors.Random(seed: 128, count: Count, dimensions: Dimensions));

        Assert.Throws<ArgumentNullException>(() => index.RemoveIds((IDSelector)null!));
    }

    private static IIDRemovableFloatIndex Removable(FloatIndex index) => Assert.IsAssignableFrom<IIDRemovableFloatIndex>(index);

    /// <summary>Adds sequentially when the row supports it (ids are positions), otherwise with ids 100..105.</summary>
    private static long[] PopulateFloat(IndexCase c, FloatIndex index, float[] vectors)
    {
        if (c.Has(IndexCapabilities.SequentialAdd))
        {
            Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Count, vectors);
            return Enumerable.Range(0, Count).Select(i => (long)i).ToArray();
        }

        long[] ids = [100, 101, 102, 103, 104, 105];
        Assert.IsAssignableFrom<IIDMappedFloatIndex>(index).Add(Count, vectors, ids);
        return ids;
    }

    private static long[] PopulateBinary(BinaryIndexCase c, BinaryIndex index, byte[] vectors)
    {
        if (c.Has(IndexCapabilities.SequentialAdd))
        {
            Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index).Add(Count, vectors);
            return Enumerable.Range(0, Count).Select(i => (long)i).ToArray();
        }

        long[] ids = [100, 101, 102, 103, 104, 105];
        Assert.IsAssignableFrom<IIDMappedBinaryIndex>(index).Add(Count, vectors, ids);
        return ids;
    }

    /// <summary>Label each source position should carry after removal.</summary>
    /// <returns><c>null</c> if not found.</returns>
    private static long?[] ExpectedLabels(ICapabilityCase c, long[] ids, int[] removed)
    {
        var expected = new long?[Count];
        long next = 0;
        for (var i = 0; i < Count; i++)
        {
            if (removed.Contains(i))
                continue;

            expected[i] = c.Has(IndexCapabilities.RemoveKeepsIds) ? ids[i] : next;
            next++;
        }

        return expected;
    }

    private static void AssertSurvivorLabels(string name, MetricType metric, long?[] expected, long[] labels, int k)
    {
        var survivorLabels = expected.Where(l => l.HasValue).Select(l => l!.Value).Order().ToArray();

        for (var q = 0; q < Count; q++)
        {
            var row = labels.AsSpan(q * k, k).ToArray();
            Assert.Equal(survivorLabels, row.Order().ToArray());

            if (metric != MetricType.InnerProduct && expected[q] is long own)
                Assert.True(own == row[0], $"{name}: source vector {q} should be found first as label {own}, got {row[0]}.");
        }
    }

    private static TheoryData<TCase, SelectorCase> Cross<TCase>(TheoryData<TCase> cases)
    {
        var data = new TheoryData<TCase, SelectorCase>();
        foreach (var row in cases)
        {
            foreach (var selector in Selectors)
            {
                data.Add(row.Data, selector);
            }
        }

        return data;
    }
}
