using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Serializer;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

/// <summary>
/// Clone, deserialization and the factory hand back a native object that already exists, with its ownership flags
/// already set by faiss (always <c>own_fields = true</c> for clone and read). Thus wrapping it must not create anything
/// or write any flag. The managed wrapper adopts the native state, and children it exposes are non-owning views.
/// </summary>
public class WrappingContractTests
{
    private const int Dimensions = 4;
    private const int Count = 8;

    private static readonly float[] Data = Vectors.Random(seed: 501, count: Count, dimensions: Dimensions);
    private static readonly long[] Ids = [100, 101, 102, 103, 104, 105, 106, 107];

    /// <param name="Build">An owning, populated index whose child native frees.</param>
    /// <param name="FirstLabel">The label a self-query for the first stored vector must return.</param>
    public sealed record WrapCase(string Name, Func<FloatIndex> Build, Func<FloatIndex, FloatIndex> CloneOf, Func<Stream, FloatIndex> Load, Func<FloatIndex, bool> ReadFlag, long FirstLabel)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<WrapCase> Cases { get; } =
    [
        new WrapCase("IVFFlat", BuildIvfFlat,
            i => ((IndexIVFFlat<IndexFlatL2>)i).Clone(),
            s => IndexDeserializer.Read<IndexIVFFlat<IndexFlatL2>>(s),
            i => ((IndexIVFFlat<IndexFlatL2>)i).OwnQuantizer,
            FirstLabel: 0),
        new WrapCase("IDMap", BuildIdMap,
            i => ((IndexIDMap<IndexFlatL2>)i).Clone(),
            s => IndexDeserializer.Read<IndexIDMap<IndexFlatL2>>(s),
            i => ((IndexIDMap<IndexFlatL2>)i).OwnsSubIndex,
            FirstLabel: 100),
        new WrapCase("IDMap2", BuildIdMap2,
            i => ((IndexIDMap2<IndexFlatL2>)i).Clone(),
            s => IndexDeserializer.Read<IndexIDMap2<IndexFlatL2>>(s),
            i => ((IndexIDMap2<IndexFlatL2>)i).OwnsSubIndex,
            FirstLabel: 100),
        new WrapCase("PreTransform", BuildPreTransform,
            i => ((IndexPreTransform<IndexFlatL2>)i).Clone(),
            s => IndexDeserializer.Read<IndexPreTransform<IndexFlatL2>>(s),
            i => ((IndexPreTransform<IndexFlatL2>)i).OwnSubIndex,
            FirstLabel: 0),
    ];

    private static FloatIndex BuildIvfFlat()
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
        index.TrainAsync(80, Vectors.Random(seed: 502, count: 80, dimensions: Dimensions)).GetAwaiter().GetResult();
        index.Nprobe = index.Nlist;
        index.Add(Count, Data);
        return index;
    }

    private static FloatIndex BuildIdMap()
    {
        var index = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);
        index.Add(Count, Data, Ids);
        return index;
    }

    private static FloatIndex BuildIdMap2()
    {
        var index = new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);
        index.Add(Count, Data, Ids);
        return index;
    }

    private static FloatIndex BuildPreTransform()
    {
        var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        index.Add(Count, Data);
        return index;
    }

    private static FloatIndex RoundTrip(WrapCase c, FloatIndex original)
    {
        using var stream = new MemoryStream();
        IndexSerializer.Write((ISerializableFloatIndex)original, stream);
        stream.Position = 0;
        return c.Load(stream);
    }

    /// <summary>faiss sets <c>own_fields = true</c> on every clone; a wrapper that overwrites it reports something else.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Clone_KeepsTheNativeOwnershipFlag(WrapCase c)
    {
        using var original = c.Build();
        using var clone = c.CloneOf(original);

        Assert.True(c.ReadFlag(clone));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Deserialize_KeepsTheNativeOwnershipFlag(WrapCase c)
    {
        using var original = c.Build();
        using var restored = RoundTrip(c, original);

        Assert.True(c.ReadFlag(restored));
    }

    /// <summary>The wrapped copy must be the index itself, not a new object layered on top of it.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void WrappedCopies_SearchLikeTheOriginal(WrapCase c)
    {
        using var original = c.Build();
        using var clone = c.CloneOf(original);
        using var restored = RoundTrip(c, original);

        foreach (var copy in new[] { clone, restored })
        {
            Assert.Equal(Count, copy.TotalCount);

            var distances = new float[1];
            var labels = new long[1];
            copy.Search(1, Data.AsSpan(0, Dimensions), k: 1, distances, labels);

            Assert.Equal(c.FirstLabel, labels[0]);
            FloatAssert.Equal(0f, distances[0]);
        }
    }

    /// <summary>
    /// Exactly one owner per native object: disposed copies and copies left to the finalizer must both free once.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void WrappedCopies_FreeExactlyOnce_UnderGcPressure(WrapCase c)
    {
        for (var i = 0; i < 20; i++)
        {
            using (var original = c.Build())
            {
                c.CloneOf(original).Dispose();
                RoundTrip(c, original).Dispose();

                // Never disposed: only finalizers run.
                _ = c.CloneOf(original).TotalCount;
                _ = RoundTrip(c, original).TotalCount;
            }
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// The factory returns an existing native IndexIDMap.
    /// </summary>
    [Fact]
    public void Factory_IdMap_IsTheMapItself()
    {
        using var index = IndexFactory.Create<IndexIDMap<IndexFlatL2>>("IDMap,Flat", Dimensions);
        index.Add(Count, Data, Ids);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, Data.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(Count, index.TotalCount);
        Assert.Equal(100, labels[0]);
        Assert.True(index.OwnsSubIndex);
    }

    [Fact]
    public void Factory_IdMap2_IsTheMapItself()
    {
        using var index = IndexFactory.Create<IndexIDMap2<IndexFlatL2>>("IDMap2,Flat", Dimensions);
        index.Add(Count, Data, Ids);

        Assert.Equal(Count, index.TotalCount);
        FloatAssert.Equal(Data.AsSpan(Dimensions, Dimensions), index.Reconstruct(101));
        Assert.True(index.OwnsSubIndex);
    }
}
