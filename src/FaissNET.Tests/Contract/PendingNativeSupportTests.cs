using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Refine;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Serializer;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

/// <summary>
/// Contracts for index types whose bindings are complete but whose C API entry points the shipped
/// <c>libfaiss_c</c> does not export yet. The missing bindings are submitted as PRs to faiss upstream
/// but not merged yet. See README for missing PRs.
/// </summary>
public class PendingNativeSupportTests
{
    private const int Count = 8;

    [Theory(Skip = NativeSupport.ReplicasOwnIndices)]
    [MemberData(nameof(IndexCatalog.PendingFloat), MemberType = typeof(IndexCatalog))]
    public void Create_IsEmptyWithExpectedShape(IndexCase c)
    {
        using var index = c.CreateIndex();

        Assert.Equal(c.Dimensions, index.Dimensions);
        Assert.Equal(c.Metric, index.Metric);
        Assert.Equal(0, index.TotalCount);
    }

    [Theory(Skip = NativeSupport.ReplicasOwnIndices)]
    [MemberData(nameof(IndexCatalog.PendingFloat), MemberType = typeof(IndexCatalog))]
    public void Add_ThenSearch_ReturnsAddedVectors(IndexCase c)
    {
        const int k = 3;

        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 50, count: Count, dimensions: c.Dimensions);
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Count, vectors);

        Assert.Equal(Count, index.TotalCount);

        var distances = new float[k];
        var labels = new long[k];
        index.Search(1, vectors.AsSpan(0, c.Dimensions), k, distances, labels);

        // Quantized storage may reorder ties, but the query's own vector must still come back first.
        Assert.Equal(0, labels[0]);
        Assert.All(labels, label => Assert.InRange(label, 0L, Count - 1L));
    }

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
    public void Replicas_AddFansOutToEveryReplica()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));

        var vectors = Vectors.Random(seed: 53, count: Count, dimensions: dimensions);
        replicas.Add(Count, vectors);

        Assert.Equal(Count, replicas.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        replicas.Search(1, vectors.AsSpan(0, dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
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

    /// <summary>Removal is only legal when the caller kept ownership; see <see cref="Replicas_RemoveReplica_RejectedWhileNativeOwnsTheReplica"/>.</summary>
    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
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

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
    public void Replicas_RemoveReplica_RejectedWhileNativeOwnsTheReplica()
    {
        const int dimensions = 4;

        using var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        var replica = new IndexFlatL2(dimensions);
        replicas.AddReplica(replica);

        Assert.True(replicas.OwnIndices);

        Assert.Throws<InvalidOperationException>(() => replicas.RemoveReplica(replica));
    }

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
    public void Replicas_DimensionMismatch_Throws()
    {
        using var replicas = new IndexReplicas<IndexFlatL2>(4);
        replicas.AddReplica(new IndexFlatL2(4));

        var mismatched = new IndexFlatL2(8);

        Assert.Throws<ArgumentException>(() => replicas.AddReplica(mismatched));
    }

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
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

    [Theory(Skip = NativeSupport.ReplicasOwnIndices)]
    [InlineData(true)]
    [InlineData(false)]
    public void Replicas_ViewOfAnotherIndex_AsReplica_IsRefused(bool ownIndices)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(4), takeOwnership: true);
        using var replicas = new IndexReplicas<IndexFlatL2>(4, ownIndices: ownIndices);

        Assert.Throws<ArgumentException>(() => replicas.AddReplica(map.SubIndex));
        Assert.Equal(4, map.SubIndex.Dimensions);
    }

    [Theory(Skip = NativeSupport.RefineFlatBaseIndex)]
    [InlineData(1f)]
    [InlineData(4f)]
    [InlineData(1000f)]
    public void RefineFlat_KFactor_SurvivesSerialization(float kFactor)
    {
        const int dimensions = 8;

        using var original = new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(dimensions));
        original.KFactor = kFactor;
        original.Add(4, Vectors.Random(seed: 404, count: 4, dimensions: dimensions));

        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;

        using var restored = IndexDeserializer.Read<IndexRefineFlat<IndexFlatL2>>(stream);

        Assert.Equal(kFactor, restored.KFactor);
    }

    [Fact(Skip = NativeSupport.RefineFlatBaseIndex)]
    public void RefineFlat_CloneAndDeserialize_DisposeWithoutDoubleFree()
    {
        const int dimensions = 8;

        for (var i = 0; i < 20; i++)
        {
            using var original = new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(dimensions));
            original.Add(4, Vectors.Random(seed: 407, count: 4, dimensions: dimensions));

            original.Clone().Dispose();

            using var stream = new MemoryStream();
            IndexSerializer.Write(original, stream);
            stream.Position = 0;
            IndexDeserializer.Read<IndexRefineFlat<IndexFlatL2>>(stream).Dispose();
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// With the default ownIndices = true, AddReplica invalidates each replica's managed handle. The
    /// trained-state check then reads IsTrained on the first replica through that invalidated handle,
    /// so adding a second trainable replica is expected to throw ObjectDisposedException rather than succeed.
    /// </summary>
    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
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

    [Fact(Skip = NativeSupport.ReplicasOwnIndices)]
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
