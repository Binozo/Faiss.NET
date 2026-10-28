using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Refine;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Transforms;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class OwnershipContractTests
{
    private const int Dimensions = 4;

    public sealed record OwnershipCase(string Name, Func<IndexFlatL2, bool, FloatIndex> Attach, Func<FloatIndex, bool> ReadFlag)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<OwnershipCase> Cases { get; } =
    [
        new OwnershipCase("IDMap", (child, own) => new IndexIDMap<IndexFlatL2>(child, takeOwnership: own), p => ((IndexIDMap<IndexFlatL2>)p).OwnsSubIndex),
        new OwnershipCase("IDMap2", (child, own) => new IndexIDMap2<IndexFlatL2>(child, takeOwnership: own), p => ((IndexIDMap2<IndexFlatL2>)p).OwnsSubIndex),
        new OwnershipCase("PreTransform", (child, own) => new IndexPreTransform<IndexFlatL2>(child, ownSubIndex: own), p => ((IndexPreTransform<IndexFlatL2>)p).OwnSubIndex),
        new OwnershipCase("IVFFlatQuantizer", (child, own) => new IndexIVFFlat<IndexFlatL2>(child, Dimensions, nlist: 2, ownQuantizer: own), p => ((IndexIVFFlat<IndexFlatL2>)p).OwnQuantizer),
        new OwnershipCase("Shards", Shard, p => ((IndexShards)p).OwnIndices),

        new OwnershipCase("IVFSQQuantizer",
            (child, own) => new IndexIVFScalarQuantizer<IndexFlatL2>(child, Dimensions, nlist: 2, QuantizerType.QT_8bit, ownQuantizer: own),
            p => ((IndexIVFScalarQuantizer<IndexFlatL2>)p).OwnQuantizer),
        // Build path only as PR #5514 and #5511 is missing
        new OwnershipCase("RefineFlatBase",
            (child, own) => new IndexRefineFlat<IndexFlatL2>(child, ownBaseIndex: own),
            p => ((IndexRefineFlat<IndexFlatL2>)p).OwnBaseIndex),
        new TheoryDataRow<OwnershipCase>(new OwnershipCase("Replicas", Replicate, p => ((IndexReplicas<IndexFlatL2>)p).OwnIndices))
            { Skip = NativeSupport.ReplicasOwnIndices },
    ];

    private static FloatIndex Shard(IndexFlatL2 child, bool own)
    {
        var shards = new IndexShards(Dimensions, ownIndices: own);
        shards.AddIndex(child);
        return shards;
    }

    private static FloatIndex Replicate(IndexFlatL2 child, bool own)
    {
        var replicas = new IndexReplicas<IndexFlatL2>(Dimensions, ownIndices: own);
        replicas.AddReplica(child);
        return replicas;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void OwningFlag_ReportsWhatConstructionRequested(OwnershipCase c)
    {
        var notOwnedChild = new IndexFlatL2(Dimensions);
        using (var notOwning = c.Attach(notOwnedChild, false))
        {
            Assert.False(c.ReadFlag(notOwning));
        }

        notOwnedChild.Dispose();

        var ownedChild = new IndexFlatL2(Dimensions);
        using var owning = c.Attach(ownedChild, true);

        Assert.True(c.ReadFlag(owning));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void NotOwning_ChildOutlivesTheParentAndStaysUsable(OwnershipCase c)
    {
        using var child = new IndexFlatL2(Dimensions);
        var parent = c.Attach(child, false);

        parent.Dispose();

        GC.Collect();
        GC.WaitForPendingFinalizers();

        child.Add(2, Vectors.Random(seed: 11, count: 2, dimensions: Dimensions));

        Assert.Equal(2, child.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Owning_DisposingBothSidesIsSafeUnderGcPressure(OwnershipCase c)
    {
        for (var i = 0; i < 20; i++)
        {
            var child = new IndexFlatL2(Dimensions);
            var parent = c.Attach(child, true);

            parent.Dispose();
            child.Dispose();
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Owning_NeverDisposed_IsSafeUnderGcPressure(OwnershipCase c)
    {
        for (var i = 0; i < 50; i++)
        {
            AttachAndAbandon(c);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    private static void AttachAndAbandon(OwnershipCase c) => _ = c.Attach(new IndexFlatL2(Dimensions), true).TotalCount;

    [Theory]
    [MemberData(nameof(Cases))]
    public void Owning_HandedOverReference_IsClosedImmediately(OwnershipCase c)
    {
        var child = new IndexFlatL2(Dimensions);
        using var parent = c.Attach(child, true);

        Assert.Throws<ObjectDisposedException>(() => child.TotalCount);
    }

    [Fact]
    public void PreTransform_PrependedTransformFollowsTheOwningFlag()
    {
        using var owned = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        owned.PrependTransform(new NormalizationTransform(Dimensions));

        Assert.True(owned.OwnSubIndex);
        Assert.Single(owned.TransformChain);

        using var child = new IndexFlatL2(Dimensions);
        using var borrowed = new IndexPreTransform<IndexFlatL2>(child, ownSubIndex: false);
        using var transform = new NormalizationTransform(Dimensions);
        borrowed.PrependTransform(transform);

        Assert.False(borrowed.OwnSubIndex);
        Assert.Single(borrowed.TransformChain);

        // The transform was never handed over, so the caller can still dispose it independently
        transform.Dispose();
    }

    [Fact]
    public void Shards_RemoveShard_RejectedWhileNativeOwnsTheShard()
    {
        using var shards = new IndexShards(Dimensions);
        var shard = new IndexFlatL2(Dimensions);
        shards.AddIndex(shard);

        Assert.True(shards.OwnIndices);
        Assert.Throws<InvalidOperationException>(() => shards.RemoveShard(shard));
    }

    [Fact]
    public void Shards_RemoveShard_AllowedWhenTheCallerKeptOwnership()
    {
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        using var shard = new IndexFlatL2(Dimensions);
        shards.AddIndex(shard);

        shards.RemoveShard(shard);

        shard.Add(1, Vectors.Random(seed: 12, count: 1, dimensions: Dimensions));

        Assert.Equal(1, shard.TotalCount);
    }

    private static void AssertShardsFindFirstVector(IndexShards shards, float[] data)
    {
        var distances = new float[1];
        var labels = new long[1];
        shards.Search(1, data.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Fact]
    public void Shards_NotOwned_CallerDisposesShardsEarly_ShardsKeepWorking()
    {
        var data = Vectors.Random(seed: 13, count: 4, dimensions: Dimensions);
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        var first = new IndexFlatL2(Dimensions);
        var second = new IndexFlatL2(Dimensions);
        shards.AddIndex(first);
        shards.AddIndex(second);
        shards.Add(4, data);

        first.Dispose();
        second.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Equal(4, shards.TotalCount);
        AssertShardsFindFirstVector(shards, data);
    }

    [Fact]
    public void Shards_NotOwned_RemovedShardCanBeDisposed_RemainingShardsKeepWorking()
    {
        var data = Vectors.Random(seed: 14, count: 4, dimensions: Dimensions);
        using var shards = new IndexShards(Dimensions, successiveIds: false, ownIndices: false);
        using var kept = new IndexFlatL2(Dimensions);
        var removed = new IndexFlatL2(Dimensions);
        shards.AddIndex(kept);
        shards.AddIndex(removed);
        kept.Add(4, data);

        shards.RemoveShard(removed);
        removed.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Throws<ObjectDisposedException>(() => removed.TotalCount);
        AssertShardsFindFirstVector(shards, data);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Shards_ViewOfAnotherIndex_AsShard_IsRefused(bool ownIndices)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);
        using var shards = new IndexShards(Dimensions, ownIndices: ownIndices);

        Assert.Throws<ArgumentException>(() => shards.AddIndex(map.SubIndex));
        Assert.Equal(Dimensions, map.SubIndex.Dimensions);
    }

    [Fact]
    public void Shards_AddThroughAViewOfNonOwningShards_IsRefused()
    {
        var shards = new IndexShards(Dimensions, ownIndices: false);
        using var map = new IndexIDMap<IndexShards>(shards, takeOwnership: true);
        var view = map.SubIndex;
        Assert.False(view.OwnIndices);

        using var shard = new IndexFlatL2(Dimensions);
        Assert.Throws<InvalidOperationException>(() => view.AddIndex(shard));
        Assert.Equal(0, shard.TotalCount);
    }
}
