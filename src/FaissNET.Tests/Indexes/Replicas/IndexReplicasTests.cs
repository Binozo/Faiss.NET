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
}
