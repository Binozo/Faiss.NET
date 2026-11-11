using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Replicas;

public class IndexReplicasTests
{
    private const int VectorCount = 8;

    [Fact]
    public void Replicas_AddFansOutToEveryReplica()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));

        var vectors = Vectors.Random(seed: 53, count: VectorCount, dimensions: dimensions);
        replicas.Add(VectorCount, vectors);

        Assert.Equal(VectorCount, replicas.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        replicas.Search(1, vectors.AsSpan(0, dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Fact]
    public void Replicas_Reconstruct_ReturnsAddedVectors()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));

        var vectors = Vectors.Random(seed: 54, count: 2, dimensions: dimensions);
        replicas.Add(2, vectors);

        FloatAssert.Equal(vectors.AsSpan(0, dimensions), replicas.Reconstruct(0));
        FloatAssert.Equal(vectors.AsSpan(dimensions, dimensions), replicas.Reconstruct(1));
    }

    [Fact]
    public void Replicas_RemoveReplica_LeavesRemainingReplicaSearchable()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        using var first = new IndexFlatL2(dimensions);
        using var second = new IndexFlatL2(dimensions);
        replicas.AddReplica(first);
        replicas.AddReplica(second);

        var vectors = Vectors.Random(seed: 55, count: 4, dimensions: dimensions);
        replicas.Add(4, vectors);

        replicas.RemoveReplica(second);

        var distances = new float[1];
        var labels = new long[1];
        replicas.Search(1, vectors.AsSpan(0, dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);

        Assert.Equal(4, second.TotalCount);
    }

    [Fact]
    public void Replicas_RemoveReplica_RejectedWhileNativeOwnsTheReplica()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        var replica = new IndexFlatL2(dimensions);
        replicas.AddReplica(replica);

        Assert.True(replicas.OwnIndices);

        Assert.Throws<InvalidOperationException>(() => replicas.RemoveReplica(replica));
    }

    [Fact]
    public void Replicas_DimensionMismatch_Throws()
    {
        using var replicas = new IndexReplicas<IndexFlatL2>(4);
        replicas.AddReplica(new IndexFlatL2(4));

        var mismatched = new IndexFlatL2(8);

        Assert.Throws<ArgumentException>(() => replicas.AddReplica(mismatched));
    }

    [Fact]
    public void Replicas_Count_FollowsAddAndRemove()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        using var first = new IndexFlatL2(dimensions);
        using var second = new IndexFlatL2(dimensions);

        Assert.Equal(0, replicas.Count);

        replicas.AddReplica(first);
        replicas.AddReplica(second);
        Assert.Equal(2, replicas.Count);

        replicas.RemoveReplica(first);
        Assert.Equal(1, replicas.Count);
    }

    [Fact]
    public void Replicas_VectorCountMismatch_IsRejectedAndSquadKeepsWorking()
    {
        const int dimensions = 4;
        var vectors = Vectors.Random(seed: 57, count: 4, dimensions: dimensions);

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        using var first = new IndexFlatL2(dimensions);
        replicas.AddReplica(first);
        replicas.Add(4, vectors);

        using var empty = new IndexFlatL2(dimensions);

        Assert.Throws<ArgumentException>(() => replicas.AddReplica(empty));
        Assert.Equal(1, replicas.Count);

        var distances = new float[1];
        var labels = new long[1];
        replicas.Search(1, vectors.AsSpan(0, dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Fact]
    public void Replicas_NotOwned_CallerDisposesReplicasEarly_SquadKeepsWorking()
    {
        const int dimensions = 4;
        var vectors = Vectors.Random(seed: 56, count: 4, dimensions: dimensions);

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        var first = new IndexFlatL2(dimensions);
        var second = new IndexFlatL2(dimensions);
        replicas.AddReplica(first);
        replicas.AddReplica(second);
        replicas.Add(4, vectors);

        first.Dispose();
        second.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var distances = new float[1];
        var labels = new long[1];
        replicas.Search(1, vectors.AsSpan(0, dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Replicas_ViewOfAnotherIndex_AsReplica_IsRefused(bool ownIndices)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(4), takeOwnership: true);
        using var replicas = new IndexReplicas<IndexFlatL2>(4, ownIndices: ownIndices);

        Assert.Throws<ArgumentException>(() => replicas.AddReplica(map.SubIndex));
        Assert.Equal(4, map.SubIndex.Dimensions);
    }

    [Fact]
    public async Task Replicas_SecondTrainedReplica_IsAccepted()
    {
        const int dimensions = 4;

        using var quantizerA = new IndexFlatL2(dimensions);
        using var quantizerB = new IndexFlatL2(dimensions);
        var a = new IndexIVFFlat<IndexFlatL2>(quantizerA, dimensions, nlist: 2);
        var b = new IndexIVFFlat<IndexFlatL2>(quantizerB, dimensions, nlist: 2);

        var train = Vectors.Random(seed: 408, count: 80, dimensions: dimensions);
        await a.TrainAsync(80, train);
        await b.TrainAsync(80, train);

        using var replicas = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions);
        replicas.AddReplica(a);
        replicas.AddReplica(b);

        Assert.True(replicas.IsTrained);
    }

    [Fact]
    public async Task Replicas_TrainedStateMismatch_IsRejectedInBothDirections()
    {
        const int dimensions = 4;
        var train = Vectors.Random(seed: 409, count: 80, dimensions: dimensions);

        using var q1 = new IndexFlatL2(dimensions);
        using var q2 = new IndexFlatL2(dimensions);
        using var q3 = new IndexFlatL2(dimensions);
        using var q4 = new IndexFlatL2(dimensions);

        // Untrained squad, trained newcomer.
        using (var squad = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions, ownIndices: false))
        using (var untrained = new IndexIVFFlat<IndexFlatL2>(q1, dimensions, nlist: 2))
        using (var trained = new IndexIVFFlat<IndexFlatL2>(q2, dimensions, nlist: 2))
        {
            await trained.TrainAsync(80, train);
            squad.AddReplica(untrained);
            Assert.Throws<ArgumentException>(() => squad.AddReplica(trained));
        }

        // Trained squad, untrained newcomer.
        using (var squad = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions, ownIndices: false))
        using (var trained = new IndexIVFFlat<IndexFlatL2>(q3, dimensions, nlist: 2))
        using (var untrained = new IndexIVFFlat<IndexFlatL2>(q4, dimensions, nlist: 2))
        {
            await trained.TrainAsync(80, train);
            squad.AddReplica(trained);
            Assert.Throws<FaissUntrainedException>(() => squad.AddReplica(untrained));
        }
    }

    [Fact]
    public void Indexer_ReturnsOneViewPerReplica()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.Add(VectorCount, Vectors.Random(seed: 60, count: VectorCount, dimensions: dimensions));

        Assert.Same(replicas[0], replicas[0]);
        Assert.NotSame(replicas[0], replicas[1]);
        Assert.Equal(VectorCount, replicas[1].TotalCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => replicas[-1]);
        Assert.Throws<ArgumentOutOfRangeException>(() => replicas[2]);
    }

    [Fact]
    public void Indexer_ViewDisposedByCaller_IsReplacedAndSquadKeepsWorking()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.Add(VectorCount, Vectors.Random(seed: 61, count: VectorCount, dimensions: dimensions));

        var first = replicas[0];
        first.Dispose();
        var second = replicas[0];

        Assert.NotSame(first, second);
        Assert.Equal(VectorCount, second.TotalCount);
        Assert.Equal(VectorCount, replicas.TotalCount);
    }

    [Fact]
    public void Dispose_DisposesViewsHeldByTheCaller()
    {
        const int dimensions = 4;
        var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        var view = replicas[0];

        replicas.Dispose();

        Assert.Throws<ObjectDisposedException>(() => view.TotalCount);
        Assert.Throws<ObjectDisposedException>(() => replicas[0]);
    }

    [Fact]
    public void RemoveReplica_DisposesTheViewOfTheRemovedReplica()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        var removed = new IndexFlatL2(dimensions);
        using var kept = new IndexFlatL2(dimensions);
        replicas.AddReplica(removed);
        replicas.AddReplica(kept);
        var removedView = replicas[0];
        var keptView = replicas[1];

        replicas.RemoveReplica(removed);
        removed.Dispose();

        Assert.Throws<ObjectDisposedException>(() => removedView.TotalCount);
        Assert.Same(keptView, replicas[0]);
    }

    /// <summary>faiss clears the squad's own trained flag on reset although every replica stays trained.</summary>
    [Fact]
    public async Task Reset_KeepsReplicasTrainedAndAcceptsTrainedReplicas()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions);
        replicas.AddReplica(await TrainedIvf(dimensions, seed: 910));
        replicas.Add(4, Vectors.Random(seed: 911, count: 4, dimensions: dimensions));

        replicas.Reset();

        Assert.Equal(0, replicas.TotalCount);
        Assert.True(replicas.IsTrained);
        replicas.AddReplica(await TrainedIvf(dimensions, seed: 912));
        Assert.Equal(2, replicas.Count);
    }

    [Fact]
    public async Task Reset_StillRejectsUntrainedReplicas()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions, ownIndices: false);
        using var trained = await TrainedIvf(dimensions, seed: 913);
        using var untrained = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 2, ownQuantizer: true);
        replicas.AddReplica(trained);

        replicas.Reset();

        Assert.Throws<FaissUntrainedException>(() => replicas.AddReplica(untrained));
        Assert.Equal(1, replicas.Count);
    }

    [Fact]
    public void IsTrained_IsFalseWithoutReplicas()
    {
        using var replicas = new IndexReplicas<IndexFlatL2>(4);

        Assert.False(replicas.IsTrained);
    }

    [Fact]
    public async Task IsTrained_FollowsTheReplicas()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexIVFFlat<IndexFlatL2>>(dimensions);
        replicas.AddReplica(new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 2, ownQuantizer: true));

        Assert.False(replicas.IsTrained);

        await replicas.TrainAsync(64, Vectors.Random(seed: 914, count: 64, dimensions: dimensions));

        Assert.True(replicas.IsTrained);
        Assert.True(replicas[0].IsTrained);
    }

    [Fact]
    public void RemoveReplica_LastOne_LeavesSquadUntrainedAndShiftsTheIndexer()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions, ownIndices: false);
        using var first = new IndexFlatL2(dimensions);
        using var second = new IndexFlatL2(dimensions);
        replicas.AddReplica(first);
        replicas.AddReplica(second);
        var secondView = replicas[1];

        replicas.RemoveReplica(first);

        Assert.True(replicas.IsTrained);
        Assert.Same(secondView, replicas[0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => replicas[1]);

        replicas.RemoveReplica(second);

        Assert.Equal(0, replicas.Count);
        Assert.False(replicas.IsTrained);
    }

    [Fact]
    public void Replicas_WithIdKeyedMembers_TakeIdsAsKeys_AndRejectUnknownKeysWithoutAborting()
    {
        const int dimensions = 4;
        var vectors = Vectors.Random(seed: 56, count: 2, dimensions: dimensions);
        var member = new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(dimensions));
        member.Add(2, vectors, [100, 101]);

        using var replicas = new IndexReplicas<IndexIDMap2<IndexFlatL2>>(dimensions);
        replicas.AddReplica(member);

        var residuals = new float[2 * dimensions];
        replicas.ComputeResidual(vectors, residuals, [100, 101]);
        FloatAssert.Equal(new float[2 * dimensions], residuals);
        FloatAssert.Equal(vectors, replicas.Reconstruct(100, 2));

        // Positions 0 and 1 hold vectors but are not ids: faiss throws for them inside an OpenMP loop when batched.
        Assert.Throws<FaissNativeException>(() => replicas.ComputeResidual(vectors, residuals, [0, 1]));
        Assert.Throws<FaissNativeException>(() => replicas.Reconstruct(0, 2));
    }

    [Fact]
    public void Replicas_AddWithIds_ReachesEveryReplica_AndSearchReturnsTheIds()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexIDMap2<IndexFlatL2>>(dimensions);
        replicas.AddReplica(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(dimensions)));
        replicas.AddReplica(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(dimensions)));
        var vectors = Vectors.Random(seed: 57, count: 3, dimensions: dimensions);

        replicas.Add(3, vectors, [100, 101, 102]);

        Assert.Equal(3, replicas.TotalCount);
        Assert.Equal([3L, 3L], [replicas[0].TotalCount, replicas[1].TotalCount]);
        var labels = new long[3];
        replicas.Search(3, vectors, 1, new float[3], labels);
        Assert.Equal([100L, 101L, 102L], labels);
        FloatAssert.Equal(vectors.AsSpan(dimensions, dimensions), replicas.Reconstruct(101));
        FloatAssert.Equal(vectors.AsSpan(dimensions, dimensions), replicas[1].Reconstruct(101));
    }

    [Fact]
    public void Replicas_AddWithIds_ReplicasWithoutIdSupport_ThrowNativeAndAddNothing()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));

        Assert.Throws<FaissNativeException>(() => replicas.Add(2, Vectors.Random(seed: 58, count: 2, dimensions: dimensions), [100, 101]));
        Assert.Equal([0L, 0L], [replicas[0].TotalCount, replicas[1].TotalCount]);
    }

    [Fact]
    public void Replicas_RemoveIds_RemovesFromEveryReplicaAndKeepsTheOtherIds()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexIDMap2<IndexFlatL2>>(dimensions);
        replicas.AddReplica(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(dimensions)));
        replicas.AddReplica(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(dimensions)));
        var vectors = Vectors.Random(seed: 59, count: 3, dimensions: dimensions);
        replicas.Add(3, vectors, [100, 101, 102]);

        Assert.Equal(1, replicas.RemoveIds([101]));

        Assert.Equal(2, replicas.TotalCount);
        Assert.Equal([2L, 2L], [replicas[0].TotalCount, replicas[1].TotalCount]);
        var labels = new long[3];
        replicas.Search(3, vectors, 1, new float[3], labels);
        Assert.Equal(100, labels[0]);
        Assert.NotEqual(101, labels[1]);
        Assert.Equal(102, labels[2]);
        Assert.Equal(0, replicas.RemoveIds([101]));
    }

    /// <summary>Flat replicas number by position, so the vectors after a removed one move up.</summary>
    [Fact]
    public void Replicas_RemoveIds_OnFlatReplicas_ShiftsThePositions()
    {
        const int dimensions = 4;
        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));
        var vectors = Vectors.Random(seed: 60, count: 3, dimensions: dimensions);
        replicas.Add(3, vectors);

        Assert.Equal(1, replicas.RemoveIds([1]));

        Assert.Equal([2L, 2L], [replicas[0].TotalCount, replicas[1].TotalCount]);
        FloatAssert.Equal(vectors.AsSpan(2 * dimensions, dimensions), replicas.Reconstruct(1));
    }

    private static async Task<IndexIVFFlat<IndexFlatL2>> TrainedIvf(int dimensions, int seed)
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 2, ownQuantizer: true);
        await index.TrainAsync(64, Vectors.Random(seed, 64, dimensions));
        return index;
    }
}
