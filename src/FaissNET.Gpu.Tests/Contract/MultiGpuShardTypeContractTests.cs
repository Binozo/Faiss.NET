using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

/// <summary>How each <see cref="GpuShardType"/> spreads an IVF index's vectors over the shards.</summary>
[Collection(GpuTestCollection.Name)]
public class MultiGpuShardTypeContractTests
{
    private const int Stored = 64;
    private const int Dimensions = 8;
    private const int Nlist = 4;

    public static TheoryData<GpuShardType> ShardTypes { get; } = new(Enum.GetValues<GpuShardType>());

    [Fact]
    public void IdModulo_ShardHoldsTheIdsWithItsRemainder()
    {
        using var source = new IvfSource(Sequential(Stored));
        using var shards = Shard(source.Index, GpuShardType.IdModulo, 2);

        Assert.Equal(Sequential(Stored).Where(id => id % 2 == 0), Ids(shards, 0));
        Assert.Equal(Sequential(Stored).Where(id => id % 2 == 1), Ids(shards, 1));
    }

    /// <summary>The ids are added in a shuffled order: the split follows the id values, not the insertion order.</summary>
    [Fact]
    public void IdRange_ShardHoldsAContiguousIdRange()
    {
        var shuffled = Sequential(Stored).OrderBy(id => (id * 37) % Stored).ToArray();
        using var source = new IvfSource(shuffled);
        using var shards = Shard(source.Index, GpuShardType.IdRange, 2);

        Assert.Equal(Sequential(32), Ids(shards, 0));
        Assert.Equal(Sequential(Stored)[32..], Ids(shards, 1));
    }

    /// <summary>63 ids over 5 shards: shard s gets [s * 63 / 5, (s + 1) * 63 / 5), so the ranges hold 12, 13, 12, 13 and 13 ids.</summary>
    [Fact]
    public void IdRange_UnevenSplit_FollowsFaissRangeBounds()
    {
        const int stored = 63;
        using var source = new IvfSource(Sequential(stored));
        using var shards = Shard(source.Index, GpuShardType.IdRange, 5);

        int[] starts = [0, 12, 25, 37, 50, 63];
        for (var s = 0; s < 5; s++)
            Assert.Equal(Sequential(stored)[starts[s]..starts[s + 1]], Ids(shards, s));
    }

    /// <summary>faiss asserts every id range is full, which only holds for the ids 0..ntotal-1; anything else would abort.</summary>
    [Theory]
    [InlineData("Offset")]
    [InlineData("Sparse")]
    [InlineData("Duplicate")]
    [InlineData("Negative")]
    public void IdRange_IdsOtherThanZeroToCount_ThrowArgumentAndKeepSource(string kind)
    {
        var ids = kind switch
        {
            "Offset" => Sequential(Stored).Select(id => id + 100).ToArray(),
            "Sparse" => Sequential(Stored).Select(id => id * 2).ToArray(),
            "Duplicate" => [0, .. Sequential(Stored - 1)],
            "Negative" => Sequential(Stored).Select(id => id - 1).ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        using var source = new IvfSource(ids);
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { ShardType = GpuShardType.IdRange };

        var error = Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], source.Index, options));

        Assert.Equal("cpuIndex", error.ParamName);

        // A stored id of -1 is indistinguishable from padding in search results, so the source is checked by its count.
        Assert.Equal(Stored, source.Index.TotalCount);
    }

    [Fact]
    public void IdRange_EmptySource_Shards()
    {
        using var source = new IvfSource([]);
        using var shards = Shard(source.Index, GpuShardType.IdRange, 2);

        Assert.Equal([0L, 0L], [shards[0].TotalCount, shards[1].TotalCount]);
    }

    /// <summary>Shard s gets the inverted lists [s * nlist / n, (s + 1) * nlist / n) whole.</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(5)]
    public void InvertedList_ShardHoldsWholeListRanges(int shardCount)
    {
        var ids = Sequential(Stored);
        using var source = new IvfSource(ids);
        var lists = source.ListOf();
        using var shards = Shard(source.Index, GpuShardType.InvertedList, shardCount);

        for (var s = 0; s < shardCount; s++)
        {
            var (first, end) = (s * Nlist / shardCount, (s + 1) * Nlist / shardCount);
            Assert.Equal(ids.Where(id => lists[id] >= first && lists[id] < end), Ids(shards, s));
        }
    }

    [Theory]
    [MemberData(nameof(ShardTypes))]
    public void EveryShardType_SearchesLikeTheSourceAndKeepsIdsBackOnCpu(GpuShardType type)
    {
        // IdRange only accepts the ids 0..ntotal-1; the others get ids unrelated to positions.
        var ids = type == GpuShardType.IdRange ? Sequential(Stored) : Sequential(Stored).Select(id => 1000 + 7 * id).ToArray();
        var queries = Vectors.Random(seed: 2, count: 12, dimensions: Dimensions);
        using var source = new IvfSource(ids);
        using var shards = Shard(source.Index, type, 2);

        SearchAssert.SameResults(source.Index, shards, queries, 5);

        using var back = GpuIndexProvider.TransferToCpu(shards);
        Assert.Equal(ids.Order(), SearchAssert.StoredLabels(back, Stored));
        SearchAssert.SameResults(source.Index, back, queries, 5);
    }

    private static long[] Sequential(int count) => Enumerable.Range(0, count).Select(i => (long)i).ToArray();

    private static GpuIndexShards<GpuIndexIVFFlat<IndexFlatL2>, IndexIVFFlat<IndexFlatL2>> Shard(IndexIVFFlat<IndexFlatL2> source, GpuShardType type, int shardCount)
    {
        var providers = new ProviderSet(shardCount);
        try
        {
            using var options = new GpuMultipleClonerOptions { ShardType = type };
            return GpuIndexProvider.TransferToGpuSharded(providers.Items, new int[shardCount], source, options);
        }
        finally
        {
            providers.Dispose();
        }
    }

    private static long[] Ids(GpuIndexShards<GpuIndexIVFFlat<IndexFlatL2>, IndexIVFFlat<IndexFlatL2>> shards, int shard)
    {
        var member = shards[shard];
        return member.TotalCount == 0 ? [] : SearchAssert.StoredLabels(member, (int)member.TotalCount);
    }

    /// <summary>An exhaustive IVFFlatL2 whose coarse quantizer stays accessible, to tell which list each vector is in.</summary>
    private sealed class IvfSource : IDisposable
    {
        private readonly IndexFlatL2 _quantizer = new(Dimensions);
        private readonly float[] _vectors;
        private readonly long[] _ids;

        public IvfSource(long[] ids)
        {
            _ids = ids;
            _vectors = Vectors.Random(seed: 1, count: ids.Length, dimensions: Dimensions);
            Index = new IndexIVFFlat<IndexFlatL2>(_quantizer, Dimensions, Nlist, ownQuantizer: false) { Nprobe = Nlist };
            Index.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: Dimensions)).GetAwaiter().GetResult();
            if (ids.Length > 0)
                Index.Add(ids.Length, _vectors, ids);
        }

        public IndexIVFFlat<IndexFlatL2> Index { get; }

        /// <summary>The inverted list of every id: the nearest coarse centroid of its vector.</summary>
        public Dictionary<long, long> ListOf()
        {
            var (_, lists) = SearchAssert.Search(_quantizer, _vectors, 1);
            return _ids.Select((id, position) => (id, list: lists[position])).ToDictionary(x => x.id, x => x.list);
        }

        public void Dispose()
        {
            Index.Dispose();
            _quantizer.Dispose();
        }
    }
}
