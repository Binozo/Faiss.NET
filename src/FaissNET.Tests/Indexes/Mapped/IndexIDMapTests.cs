using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Mapped;

public class IndexIDMapTests
{
    private const int Dimensions = 4;
    private const int Count = 4;

    private static readonly float[] Data = Vectors.Random(seed: 601, count: Count, dimensions: Dimensions);
    private static readonly long[] Ids = [100, 101, 102, 103];

    public sealed record MapCase(string Name, Func<IndexFlatL2, bool, FloatIndex> Build, Func<FloatIndex, IndexFlatL2> SubIndexOf, Func<FloatIndex, FloatIndex> CloneOf)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<MapCase> Maps { get; } =
    [
        new MapCase("IDMap",
            (sub, own) => new IndexIDMap<IndexFlatL2>(sub, own),
            m => ((IndexIDMap<IndexFlatL2>)m).SubIndex,
            m => ((IndexIDMap<IndexFlatL2>)m).Clone()),
        new MapCase("IDMap2",
            (sub, own) => new IndexIDMap2<IndexFlatL2>(sub, own),
            m => ((IndexIDMap2<IndexFlatL2>)m).SubIndex,
            m => ((IndexIDMap2<IndexFlatL2>)m).Clone()),
    ];

    private static FloatIndex Populated(MapCase c, IndexFlatL2 sub, bool own)
    {
        var map = c.Build(sub, own);
        ((IIDMappedFloatIndex)map).Add(Count, Data, Ids);
        return map;
    }

    private static void AssertFindsFirstVector(FloatIndex map)
    {
        var distances = new float[1];
        var labels = new long[1];
        map.Search(1, Data.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(Ids[0], labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void SubIndex_Owned_IsAViewOfTheWrappedIndex(MapCase c)
    {
        var original = new IndexFlatL2(Dimensions);
        using var map = Populated(c, original, own: true);

        var sub = c.SubIndexOf(map);

        Assert.NotSame(original, sub);
        Assert.Equal(Dimensions, sub.Dimensions);
        Assert.Equal(Count, sub.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void SubIndex_Owned_ThrowsOnceTheMapIsDisposed(MapCase c)
    {
        var map = Populated(c, new IndexFlatL2(Dimensions), own: true);
        var sub = c.SubIndexOf(map);

        map.Dispose();

        Assert.Throws<ObjectDisposedException>(() => sub.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void SubIndex_NotOwned_IsTheCallersObject(MapCase c)
    {
        using var original = new IndexFlatL2(Dimensions);
        using var map = Populated(c, original, own: false);

        Assert.Same(original, c.SubIndexOf(map));
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void NotOwned_SubIndexOutlivesTheMap(MapCase c)
    {
        using var original = new IndexFlatL2(Dimensions);
        var map = Populated(c, original, own: false);

        map.Dispose();

        Assert.Equal(Count, original.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void NotOwned_CallerDisposesSubIndexEarly_MapKeepsWorking(MapCase c)
    {
        var original = new IndexFlatL2(Dimensions);
        using var map = Populated(c, original, own: false);

        original.Dispose();

        Assert.Equal(Count, map.TotalCount);
        AssertFindsFirstVector(map);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void SubIndex_OfAClone_IsAUsableView(MapCase c)
    {
        using var map = Populated(c, new IndexFlatL2(Dimensions), own: true);
        var clone = c.CloneOf(map);
        var sub = c.SubIndexOf(clone);

        Assert.Equal(Count, sub.TotalCount);
        AssertFindsFirstVector(clone);

        clone.Dispose();
        Assert.Throws<ObjectDisposedException>(() => sub.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Maps))]
    public void SubIndexOfAnotherMap_IsRefused(MapCase c)
    {
        using var first = Populated(c, new IndexFlatL2(Dimensions), own: true);
        var view = c.SubIndexOf(first);

        Assert.Throws<ArgumentException>(() => c.Build(view, true));
        Assert.Throws<ArgumentException>(() => c.Build(view, false));

        Assert.Equal(Count, view.TotalCount);
        AssertFindsFirstVector(first);
    }

    [Fact]
    public void IdMap2_Reconstruct_ByCustomId_ReturnsAddedVector()
    {
        using var sub = new IndexFlatL2(4);
        using var map = new IndexIDMap2<IndexFlatL2>(sub, takeOwnership: false);
        float[] vector = [1f, 2f, 3f, 4f];
        map.Add(1, vector, [100]);

        FloatAssert.Equal(vector, map.Reconstruct(100));
    }
}
