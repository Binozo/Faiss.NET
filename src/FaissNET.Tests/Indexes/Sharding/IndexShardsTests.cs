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
}
