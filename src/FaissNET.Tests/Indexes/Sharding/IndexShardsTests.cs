using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Sharding;
public class IndexShardsTests
{
    private const int Dimensions = 4;

    [Fact]
    public void AddIndex_DimensionMismatch_Throws()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(new IndexFlatL2(Dimensions));
        using var wrong = new IndexFlatL2(Dimensions * 2);

        Assert.Equal("index", Assert.Throws<ArgumentException>(() => shards.AddIndex(wrong)).ParamName);
        Assert.Equal(Dimensions * 2, wrong.Dimensions);
    }

    [Fact]
    public void SuccessiveIds_CustomIdsAreRefused()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(new IndexFlatL2(Dimensions));

        Assert.True(shards.SuccessiveIDs);
        Assert.Throws<InvalidOperationException>(() => shards.Add(1, new float[Dimensions], [5]));
        Assert.Equal(0, shards.TotalCount);
    }

    [Fact]
    public void CustomIds_WithIdCapableShards_AreReturnedBySearch()
    {
        using var shards = new IndexShards(Dimensions, successiveIds: false);
        shards.AddIndex(new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true));
        shards.AddIndex(new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true));
        var vectors = Vectors.Random(seed: 890, count: 4, dimensions: Dimensions);

        Assert.False(shards.SuccessiveIDs);
        shards.Add(4, vectors, [100, 101, 102, 103]);

        var labels = new long[4];
        shards.Search(4, vectors, 1, new float[4], labels);
        Assert.Equal([100L, 101L, 102L, 103L], labels);
    }

    [Fact]
    public void CustomIds_WithShardsWithoutIdSupport_Throw()
    {
        using var shards = new IndexShards(Dimensions, successiveIds: false);
        shards.AddIndex(new IndexFlatL2(Dimensions));
        var vectors = Vectors.Random(seed: 891, count: 2, dimensions: Dimensions);

        Assert.Throws<FaissNativeException>(() => shards.Add(2, vectors, [100, 101]));
        Assert.Throws<FaissNativeException>(() => shards.Add(2, vectors));
        Assert.Equal(0, shards.TotalCount);
    }

    [Fact]
    public async Task TrainAsync_TrainsEveryShard()
    {
        using var shards = new IndexShards(Dimensions);
        var first = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
        var second = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
        shards.AddIndex(first);
        shards.AddIndex(second);

        Assert.False(shards.IsTrained);
        await shards.TrainAsync(64, Vectors.Random(seed: 892, count: 64, dimensions: Dimensions));

        Assert.True(shards.IsTrained);
        shards.Add(4, Vectors.Random(seed: 893, count: 4, dimensions: Dimensions));
        Assert.Equal(4, shards.TotalCount);
    }

    [Fact]
    public void Count_FollowsAddAndRemove()
    {
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        using var first = new IndexFlatL2(Dimensions);
        using var second = new IndexFlatL2(Dimensions);

        Assert.Equal(0, shards.Count);

        shards.AddIndex(first);
        shards.AddIndex(second);
        Assert.Equal(2, shards.Count);

        shards.RemoveShard(first);
        Assert.Equal(1, shards.Count);
    }

    [Fact]
    public async Task TrainedStateMismatch_IsRejectedInBothDirections()
    {
        var train = Vectors.Random(seed: 894, count: 80, dimensions: Dimensions);

        using var q1 = new IndexFlatL2(Dimensions);
        using var q2 = new IndexFlatL2(Dimensions);
        using var q3 = new IndexFlatL2(Dimensions);
        using var q4 = new IndexFlatL2(Dimensions);

        // Untrained shards, trained newcomer.
        using (var shards = new IndexShards(Dimensions, ownIndices: false))
        using (var untrained = new IndexIVFFlat<IndexFlatL2>(q1, Dimensions, nlist: 2))
        using (var trained = new IndexIVFFlat<IndexFlatL2>(q2, Dimensions, nlist: 2))
        {
            shards.AddIndex(untrained);
            await trained.TrainAsync(80, train);

            Assert.Throws<ArgumentException>(() => shards.AddIndex(trained));
            Assert.Equal(1, shards.Count);
            Assert.False(shards.IsTrained);
        }

        // Trained shards, untrained newcomer.
        using (var shards = new IndexShards(Dimensions, ownIndices: false))
        using (var trained = new IndexIVFFlat<IndexFlatL2>(q3, Dimensions, nlist: 2))
        using (var untrained = new IndexIVFFlat<IndexFlatL2>(q4, Dimensions, nlist: 2))
        {
            await trained.TrainAsync(80, train);
            shards.AddIndex(trained);

            Assert.Throws<FaissUntrainedException>(() => shards.AddIndex(untrained));
            Assert.Equal(1, shards.Count);

            shards.Add(4, train.AsSpan(0, 4 * Dimensions));
            Assert.Equal(4, shards.TotalCount);
        }
    }

    [Fact]
    public void Indexer_ReturnsOneViewPerShard()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(Populated(count: 3, seed: 895));
        shards.AddIndex(Populated(count: 1, seed: 896));

        Assert.Same(shards[0], shards[0]);
        Assert.NotSame(shards[0], shards[1]);
        Assert.Equal(3, shards[0].TotalCount);
        Assert.Equal(1, shards[1].TotalCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => shards[-1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => shards[2]);
    }

    [Fact]
    public void Indexer_ViewDisposedByCaller_IsReplacedAndShardsKeepWorking()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(Populated(count: 3, seed: 897));

        var first = shards[0];
        first.Dispose();
        var second = shards[0];

        Assert.NotSame(first, second);
        Assert.Equal(3, second.TotalCount);
        Assert.Equal(3, shards.TotalCount);
    }

    [Fact]
    public void Dispose_DisposesViewsHeldByTheCaller()
    {
        var shards = new IndexShards(Dimensions);
        shards.AddIndex(Populated(count: 3, seed: 898));
        var view = shards[0];

        shards.Dispose();

        Assert.Throws<ObjectDisposedException>(() => view.TotalCount);
        Assert.Throws<ObjectDisposedException>(() => shards[0]);
    }

    [Fact]
    public void RemoveShard_DisposesTheViewOfTheRemovedShard()
    {
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        var removed = Populated(count: 3, seed: 899);
        using var kept = Populated(count: 1, seed: 900);
        shards.AddIndex(removed);
        shards.AddIndex(kept);
        var removedView = shards[0];
        var keptView = shards[1];

        shards.RemoveShard(removed);
        removed.Dispose();

        Assert.Throws<ObjectDisposedException>(() => removedView.TotalCount);
        Assert.Same(keptView, shards[0]);
        Assert.Equal(1, keptView.TotalCount);
    }

    /// <summary>faiss clears the shards' own trained flag on reset although every shard stays trained.</summary>
    [Fact]
    public async Task Reset_KeepsShardsTrainedAndAcceptsTrainedShards()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(await TrainedIvf(seed: 901));
        shards.Add(4, Vectors.Random(seed: 902, count: 4, dimensions: Dimensions));

        shards.Reset();

        Assert.Equal(0, shards.TotalCount);
        Assert.True(shards.IsTrained);
        shards.AddIndex(await TrainedIvf(seed: 903));
        Assert.Equal(2, shards.Count);

        shards.Add(4, Vectors.Random(seed: 904, count: 4, dimensions: Dimensions));
        Assert.Equal(4, shards.TotalCount);
    }

    [Fact]
    public async Task Reset_StillRejectsUntrainedShards()
    {
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        using var trained = await TrainedIvf(seed: 905);
        using var untrained = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
        shards.AddIndex(trained);

        shards.Reset();

        Assert.Throws<FaissUntrainedException>(() => shards.AddIndex(untrained));
        Assert.Equal(1, shards.Count);
    }

    [Fact]
    public void IsTrained_IsFalseWithoutShards()
    {
        using var shards = new IndexShards(Dimensions);

        Assert.False(shards.IsTrained);
    }

    [Fact]
    public async Task IsTrained_FollowsTheShards()
    {
        using var ivfShards = new IndexShards(Dimensions);
        ivfShards.AddIndex(new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true));
        using var flatShards = new IndexShards(Dimensions);
        flatShards.AddIndex(new IndexFlatL2(Dimensions));

        Assert.False(ivfShards.IsTrained);
        Assert.True(flatShards.IsTrained);

        await ivfShards.TrainAsync(64, Vectors.Random(seed: 906, count: 64, dimensions: Dimensions));

        Assert.True(ivfShards.IsTrained);
    }

    [Fact]
    public void RemoveShard_LastOne_LeavesShardsUntrainedAndShiftsTheIndexer()
    {
        using var shards = new IndexShards(Dimensions, ownIndices: false);
        using var first = Populated(count: 2, seed: 907);
        using var second = Populated(count: 3, seed: 908);
        shards.AddIndex(first);
        shards.AddIndex(second);

        shards.RemoveShard(first);

        Assert.Equal(3, shards[0].TotalCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => shards[1]);

        shards.RemoveShard(second);

        Assert.Equal(0, shards.Count);
        Assert.False(shards.IsTrained);
    }

    private static async Task<IndexIVFFlat<IndexFlatL2>> TrainedIvf(int seed)
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
        await index.TrainAsync(64, Vectors.Random(seed, 64, Dimensions));
        return index;
    }

    private static IndexFlatL2 Populated(int count, int seed)
    {
        var index = new IndexFlatL2(Dimensions);
        index.Add(count, Vectors.Random(seed, count, Dimensions));
        return index;
    }
}
