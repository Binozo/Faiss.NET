using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

using FlatShards = GpuIndexShards<GpuIndexFlatL2, IndexFlatL2>;
using IvfShards = GpuIndexShards<GpuIndexIVFFlat<IndexFlatL2>, IndexIVFFlat<IndexFlatL2>>;

[Collection(GpuTestCollection.Name)]
public class MultiGpuSquadOperationsContractTests
{
    private const int Stored = 64;
    private const int Dimensions = 8;
    private const int K = 5;

    [Fact]
    public void ShardedFlat_AddWithIds_ThrowsInvalidOperationAndAddsNothing()
    {
        using var cpu = FlatL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);

        Assert.Throws<InvalidOperationException>(() => shards.Add(1, new float[Dimensions], [5]));
        Assert.Equal(Stored, shards.TotalCount);
    }

    /// <summary>faiss only supports one add into flat shards with successive ids: a second would renumber the vectors.</summary>
    [Fact]
    public void ShardedFlat_AddAfterVectors_ThrowsNativeAndAddsNothing()
    {
        using var cpu = FlatL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);

        Assert.Throws<FaissNativeException>(() => shards.Add(1, new float[Dimensions]));
        Assert.Equal(Stored, shards.TotalCount);
    }

    [Fact]
    public void ShardedFlat_FirstAddIntoEmptyShards_SplitsContiguouslyWithGlobalLabels()
    {
        var vectors = Vectors.Random(seed: 3, count: 8, dimensions: Dimensions);
        using var cpu = FlatL2(0);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);

        shards.Add(8, vectors);

        Assert.Equal([4L, 4L], [shards[0].TotalCount, shards[1].TotalCount]);
        var (_, labels) = SearchAssert.Search(shards, vectors, 1);
        Assert.Equal(Enumerable.Range(0, 8).Select(i => (long)i), labels);
        Assert.Throws<FaissNativeException>(() => shards.Add(1, vectors[..Dimensions]));
    }

    [Fact]
    public void ShardedIvf_AddWithoutIds_NumbersFromTheCount()
    {
        var added = Vectors.Random(seed: 3, count: 4, dimensions: Dimensions);
        using var cpu = IvfL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);

        shards.Add(4, added);

        Assert.Equal(Stored + 4, shards.TotalCount);
        var (_, labels) = SearchAssert.Search(shards, added, 1);
        Assert.Equal([64L, 65L, 66L, 67L], labels);
    }

    [Fact]
    public void ShardedIvf_AddWithIds_ReportsAndKeepsThemOnCpu()
    {
        var added = Vectors.Random(seed: 3, count: 4, dimensions: Dimensions);
        long[] ids = [9_000_000_000, 9_000_000_013, 9_000_000_026, 9_000_000_039];
        using var cpu = IvfL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);

        shards.Add(4, added, ids);

        var (_, labels) = SearchAssert.Search(shards, added, 1);
        Assert.Equal(ids, labels);
        using var back = GpuIndexProvider.TransferToCpu(shards);
        Assert.Equal([.. Enumerable.Range(0, Stored).Select(i => (long)i), .. ids], SearchAssert.StoredLabels(back, Stored + 4));
    }

    [Fact]
    public void Replicas_Add_ReachesEveryReplica()
    {
        var added = Vectors.Random(seed: 3, count: 8, dimensions: Dimensions);
        using var cpu = IvfL2(Stored);
        using var providers = new ProviderSet(3);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0, 0], cpu);

        replicas.Add(8, added);

        Assert.Equal([72L, 72L, 72L], [replicas[0].TotalCount, replicas[1].TotalCount, replicas[2].TotalCount]);
        var (_, labels) = SearchAssert.Search(replicas, added, 1);
        Assert.Equal(Enumerable.Range(Stored, 8).Select(i => (long)i), labels);
    }

    /// <summary>Replicas split the queries in blocks of ceil(n / replicas). Idle replicas and uneven blocks must not change results.</summary>
    [Theory]
    [InlineData(1, 2)]
    [InlineData(3, 2)]
    [InlineData(5, 5)]
    [InlineData(2, 5)]
    public void Replicas_QueriesSplitAcrossReplicas_MatchTheSource(int queryCount, int replicaCount)
    {
        var queries = Vectors.Random(seed: 2, count: queryCount, dimensions: Dimensions);
        using var cpu = FlatL2(Stored);
        using var providers = new ProviderSet(replicaCount);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, new int[replicaCount], cpu);

        SearchAssert.SameResults(cpu, replicas, queries, K);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Search_NoQueries_LeavesTheBuffersUntouched(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);
        var distances = new[] { 42f };
        var labels = new[] { 42L };

        squad.Search(0, [], 1, distances, labels);

        Assert.Equal(42f, distances[0]);
        Assert.Equal(42L, labels[0]);
    }

    [Fact]
    public void ShardedIvf_SearchWithNprobe_OverridesTheShardsNprobe()
    {
        var queries = Vectors.Random(seed: 2, count: 16, dimensions: Dimensions);
        using var cpu = IvfL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);
        using var oneList = new SearchParametersIVF(nprobe: 1);

        var cpuOneList = SearchWithParams(cpu, queries, oneList);
        var shardsOneList = SearchWithParams(shards, queries, oneList);

        Assert.NotEqual(SearchAssert.Search(cpu, queries, K).Labels, cpuOneList.Labels);
        Assert.Equal(cpuOneList.Labels, shardsOneList.Labels);
        FloatAssert.Equal(cpuOneList.Distances, shardsOneList.Distances, relTol: 1e-4f, absTol: 1e-4f);
    }

    [Fact]
    public void ShardedIvf_UnsupportedSearchParameters_ThrowNative()
    {
        using var cpu = IvfL2(8);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);
        using var plain = new SearchParameters();
        using var maxCodes = new SearchParametersIVF(nprobe: 4, maxCodes: 1);

        Assert.Throws<FaissNativeException>(() => SearchWithParams(shards, new float[Dimensions], plain));
        Assert.Throws<FaissNativeException>(() => SearchWithParams(shards, new float[Dimensions], maxCodes));
    }

    [Fact]
    public void Shards_SearchWithSelector_ThrowsNotSupported()
    {
        using var flatCpu = FlatL2(8);
        using var ivfCpu = IvfL2(8);
        using var providers = new ProviderSet(2);
        using var flatShards = Sharded(providers, flatCpu);
        using var ivfShards = Sharded(providers, ivfCpu);
        using var selector = IDSelector.Range(0, 4);
        using var flatParameters = new SearchParameters(selector);
        using var ivfParameters = new SearchParametersIVF(nprobe: 4, selector: selector);

        Assert.Throws<NotSupportedException>(() => SearchWithParams(flatShards, new float[Dimensions], flatParameters));
        Assert.Throws<NotSupportedException>(() => SearchWithParams(ivfShards, new float[Dimensions], ivfParameters));
    }

    [Fact]
    public void ShardedFlat_SearchWithPlainParameters_MatchesSearch()
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: Dimensions);
        using var cpu = FlatL2(Stored);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);
        using var parameters = new SearchParameters();

        var withParameters = SearchWithParams(shards, queries, parameters);
        var plain = SearchAssert.Search(shards, queries, K);

        Assert.Equal(plain.Labels, withParameters.Labels);
        Assert.Equal(plain.Distances, withParameters.Distances);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Search_KBeyondSelectionLimit_ThrowsNative(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);
        
        // The device k-selection stops at 2048

        // The per-dimension scalar quantizer refuses earlier, in managed code (see IvfSq8_KAboveScanLimit_ThrowsBeforeNative).
        if (c.Name == "IVFSQ8")
            Assert.Throws<ArgumentOutOfRangeException>(() => SearchAssert.Search(squad, new float[c.Dimensions], 2049));
        else
            Assert.Throws<FaissNativeException>(() => SearchAssert.Search(squad, new float[c.Dimensions], 2049));

        Assert.Equal(8, squad.TotalCount);
    }

    /// <summary>The 8-bit scalar quantizer scan aborts above k = 512.</summary>
    [Fact]
    public void IvfSq8_KAboveScanLimit_ThrowsBeforeNative()
    {
        using var cpu = Sq(QuantizerType.QT_8bit, Stored);
        using var providers = new ProviderSet(2);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu);
        using var parameters = new SearchParametersIVF(nprobe: 4);

        foreach (FloatIndex squad in new FloatIndex[] { shards, replicas })
        {
            Assert.Equal("k", Assert.Throws<ArgumentOutOfRangeException>(() => SearchAssert.Search(squad, new float[Dimensions], 513)).ParamName);
            Assert.Throws<ArgumentOutOfRangeException>(() => squad.Assign(1, new float[Dimensions], 513, new long[513]));

            var (_, labels) = SearchAssert.Search(squad, new float[Dimensions], 512);
            Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), labels[..Stored].Order());
        }

        Assert.Throws<ArgumentOutOfRangeException>(() => shards.SearchWithParams(1, new float[Dimensions], 513, parameters, new float[513], new long[513]));
    }

    [Fact]
    public void IvfSqFp16_PerDimensionMemberJoins_SquadAdoptsItsKLimit()
    {
        using var fp16 = Sq(QuantizerType.QT_fp16, 0);
        using var sq8 = Sq(QuantizerType.QT_8bit, 0);
        using var providers = new ProviderSet(4);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items[..2], [0, 0], fp16);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items[2..], [0, 0], fp16);

        var (_, labels) = SearchAssert.Search(shards, new float[Dimensions], 513);
        Assert.All(labels, label => Assert.Equal(-1, label));

        shards.AddIndex(GpuIndexProvider.TransferToGpu(providers.Items[0], sq8));
        replicas.AddReplica(GpuIndexProvider.TransferToGpu(providers.Items[2], sq8));

        Assert.Throws<ArgumentOutOfRangeException>(() => SearchAssert.Search(shards, new float[Dimensions], 513));
        Assert.Throws<ArgumentOutOfRangeException>(() => SearchAssert.Search(replicas, new float[Dimensions], 513));
    }

    [Fact]
    public void Replicas_Reconstruct_ReturnsTheStoredVectorFromFlatAndThrowsForIvf()
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: Dimensions);
        using var flatCpu = new IndexFlatL2(Dimensions);
        flatCpu.Add(Stored, stored);
        using var ivfCpu = IvfL2(Stored);
        using var providers = new ProviderSet(4);
        using var flatReplicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items[..2], [0, 0], flatCpu);
        using var ivfReplicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items[2..], [0, 0], ivfCpu);

        Assert.Equal(stored[(5 * Dimensions)..(6 * Dimensions)], flatReplicas.Reconstruct(5));
        Assert.Throws<FaissNativeException>(() => ivfReplicas.Reconstruct(5));
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Reset_EmptiesEveryMemberAndAcceptsNewVectors(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        squad.Reset();

        Assert.Equal(0, squad.TotalCount);
        Assert.Equal([0L, 0L], c.MemberCounts(squad));
        Assert.True(((ITrainableFloatIndex)squad).IsTrained);
        var (_, empty) = SearchAssert.Search(squad, new float[c.Dimensions], K);
        Assert.All(empty, label => Assert.Equal(-1, label));

        ((IIDSequentialFloatIndex)squad).Add(8, Vectors.Random(seed: 3, count: 8, dimensions: c.Dimensions));

        Assert.Equal(8, squad.TotalCount);
        Assert.Equal(Enumerable.Range(0, 8).Select(i => (long)i), SearchAssert.StoredLabels(squad, 8));
    }

    [Fact]
    public void Shards_AddIndexOfAnotherKind_ThrowsArgumentAndKeepsTheShards()
    {
        using var cpu = FlatL2(8);
        using var ivfCpu = IvfL2(8);
        using var providers = new ProviderSet(3);
        using var shards = Sharded(providers, cpu);
        using var nested = Sharded(providers, cpu);
        using var otherKind = GpuIndexProvider.TransferToGpu(providers.Items[2], ivfCpu);
        using var widerCpu = new IndexFlatL2(Dimensions * 2);
        using var wider = GpuIndexProvider.TransferToGpu(providers.Items[2], widerCpu);
        using var cpuMember = new IndexFlatL2(Dimensions);

        Assert.Equal("index", Assert.Throws<ArgumentException>(() => shards.AddIndex(cpuMember)).ParamName);
        Assert.Equal("index", Assert.Throws<ArgumentException>(() => shards.AddIndex(nested)).ParamName);
        Assert.Equal("index", Assert.Throws<ArgumentException>(() => shards.AddIndex(otherKind)).ParamName);
        Assert.Equal("index", Assert.Throws<ArgumentException>(() => shards.AddIndex(wider)).ParamName);

        Assert.Equal(2, shards.Count);
        Assert.Equal(8, shards.TotalCount);
    }

    [Fact]
    public void Shards_AddIndexWithOtherMetric_ThrowsNativeAndKeepsTheShards()
    {
        using var l2 = new IndexFlat(Dimensions, MetricType.L2);
        using var ip = new IndexFlat(Dimensions, MetricType.InnerProduct);
        using var providers = new ProviderSet(3);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items[..2], [0, 0], l2);
        using var ipMember = GpuIndexProvider.TransferToGpu(providers.Items[2], ip);

        Assert.Throws<FaissNativeException>(() => shards.AddIndex(ipMember));

        Assert.Equal(2, shards.Count);
        Assert.Equal(MetricType.InnerProduct, ipMember.Metric);
    }

    /// <summary>Merging IVF shards back requires one set of centroids. A separately trained member is refused, not merged.</summary>
    [Fact]
    public void Shards_SeparatelyTrainedIvfMember_ToCpuThrowsNativeAndShardsStayUsable()
    {
        using var cpu = IvfL2(Stored);
        using var other = IvfL2(8, trainingSeed: 99);
        using var providers = new ProviderSet(3);
        using var shards = Sharded(providers, cpu);
        shards.AddIndex(GpuIndexProvider.TransferToGpu(providers.Items[2], other));

        Assert.Throws<FaissNativeException>(() => GpuIndexProvider.TransferToCpu(shards).Dispose());

        Assert.Equal(Stored + 8, shards.TotalCount);
        Assert.Equal(Stored + 8, SearchAssert.Search(shards, new float[Dimensions], Stored + 8).Labels.Count(label => label >= 0));
    }

    /// <summary>Transferred squads own their members, and faiss frees a removed member it owns.</summary>
    [Fact]
    public void TransferredSquads_RemoveMember_ThrowsInvalidOperation()
    {
        using var cpu = FlatL2(8);
        using var providers = new ProviderSet(2);
        using var shards = Sharded(providers, cpu);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu);

        Assert.Throws<InvalidOperationException>(() => shards.RemoveShard(shards[0]));
        Assert.Throws<InvalidOperationException>(() => replicas.RemoveReplica(replicas[0]));
        Assert.Equal(2, shards.Count);
        Assert.Equal(2, replicas.Count);
    }

    [Fact]
    public void Float16Options_ShardWithinFloat16Tolerance()
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: Dimensions);
        using var flat = FlatL2(Stored);
        using var ivf = IvfL2(Stored);
        using var providers = new ProviderSet(2);
        using var float16 = new GpuMultipleClonerOptions { UseFloat16 = true };
        using var float16Quantizer = new GpuMultipleClonerOptions { UseFloat16CoarseQuantizer = true };

        using var flatShards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], flat, float16);
        using var ivfShards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], ivf, float16Quantizer);

        SearchAssert.SameResults(flat, flatShards, queries, K, relTol: 2e-3f, absTol: 1e-3f);
        SearchAssert.SameResults(ivf, ivfShards, queries, K, relTol: 2e-3f, absTol: 1e-3f);
    }

    /// <summary>Reserved space is split over the shards and only applies to an empty source. The shards take vectors as usual.</summary>
    [Fact]
    public void ReserveVecs_EmptyIvfSource_ShardsTakeVectors()
    {
        var added = Vectors.Random(seed: 3, count: 16, dimensions: Dimensions);
        using var cpu = IvfL2(0);
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { ReserveVecs = 1000 };
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu, options);

        shards.Add(16, added);

        Assert.Equal([8L, 8L], [shards[0].TotalCount, shards[1].TotalCount]);
        var (_, labels) = SearchAssert.Search(shards, added, 1);
        Assert.Equal(Enumerable.Range(0, 16).Select(i => (long)i), labels);
    }

    /// <summary>
    /// The CPU parameter space forwards nprobe to every member but has no branch for device IVF indexes, so it refuses.
    /// Per-search <see cref="SearchParametersIVF"/> is the way to change nprobe on shards.
    /// </summary>
    [Fact]
    public void SetParameterNprobe_OnDeviceSquads_ThrowsNativeAndKeepsResults()
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: Dimensions);
        using var cpu = IvfL2(Stored);
        using var providers = new ProviderSet(4);
        using var shards = Sharded(providers, cpu);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items[2..], [0, 0], cpu);

        Assert.Throws<FaissNativeException>(() => shards.SetParameter("nprobe", 1));
        Assert.Throws<FaissNativeException>(() => replicas.SetParameter("nprobe", 1));

        SearchAssert.SameResults(cpu, shards, queries, K);
        SearchAssert.SameResults(cpu, replicas, queries, K);
    }

    /// <summary>
    /// Shards of an untrained source train on the same data and reach the same centroids,
    /// even on different GPUs, so they still merge back into one CPU index.
    /// </summary>
    [Fact]
    public async Task ShardsOfUntrainedSource_TrainOnDevices_AndMergeBackToCpu()
    {
        var last = FaissGpu.GetNumGpus() - 1;
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: Dimensions);
        using var cpu = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 4, ownQuantizer: true) { Nprobe = 4 };
        using var providers = new ProviderSet(2);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, last], cpu);
        Assert.False(shards.IsTrained);

        await shards.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: Dimensions));
        shards.Add(Stored, stored);

        Assert.True(shards.IsTrained);
        using var back = GpuIndexProvider.TransferToCpu(shards);
        Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), SearchAssert.StoredLabels(back, Stored));
        var (_, labels) = SearchAssert.Search(back, stored, 1);
        Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), labels);
    }

    /// <summary>Polysemous training leaves the search threshold at 0, so such an IVFPQ still shards.</summary>
    [Fact]
    public async Task IvfPq_PolysemousTraining_StillShards()
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: 16);
        using var cpu = new Faiss.Cpu.Indexes.Approximate.IndexIVFPQ(16, n: 4, m: 4, polysemy: true) { Nprobe = 4 };
        await cpu.TrainAsync(1024, Vectors.Random(seed: 98, count: 1024, dimensions: 16));
        cpu.Add(Stored, Vectors.Random(seed: 1, count: Stored, dimensions: 16));
        using var providers = new ProviderSet(2);

        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu);

        Assert.Equal([Stored / 2L, Stored / 2L], [shards[0].TotalCount, shards[1].TotalCount]);
        SearchAssert.SameResults(cpu, shards, queries, K, relTol: 1e-3f, absTol: 1e-3f);
    }

    private static IndexFlatL2 FlatL2(int count)
    {
        var index = new IndexFlatL2(Dimensions);
        if (count > 0)
            index.Add(count, Vectors.Random(seed: 1, count: count, dimensions: Dimensions));
        return index;
    }

    private static IndexIVFFlat<IndexFlatL2> IvfL2(int count, int trainingSeed = 98)
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 4, ownQuantizer: true) { Nprobe = 4 };
        index.TrainAsync(256, Vectors.Random(seed: trainingSeed, count: 256, dimensions: Dimensions)).GetAwaiter().GetResult();
        if (count > 0)
            index.Add(count, Vectors.Random(seed: 1, count: count, dimensions: Dimensions), Enumerable.Range(0, count).Select(i => (long)i).ToArray());
        return index;
    }

    private static IndexIVFScalarQuantizer<IndexFlatL2> Sq(QuantizerType type, int count)
    {
        var index = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 4, type, ownQuantizer: true) { Nprobe = 4 };
        index.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: Dimensions)).GetAwaiter().GetResult();
        if (count > 0)
            index.Add(count, Vectors.Random(seed: 1, count: count, dimensions: Dimensions), Enumerable.Range(0, count).Select(i => (long)i).ToArray());
        return index;
    }

    private static FlatShards Sharded(ProviderSet providers, IndexFlatL2 cpu) =>
        GpuIndexProvider.TransferToGpuSharded(providers.Items[..2], [0, 0], cpu);

    private static IvfShards Sharded(ProviderSet providers, IndexIVFFlat<IndexFlatL2> cpu) =>
        GpuIndexProvider.TransferToGpuSharded(providers.Items[..2], [0, 0], cpu);

    private static (float[] Distances, long[] Labels) SearchWithParams(IParamsFloatSearchIndex index, float[] queries, SearchParameters parameters)
    {
        var count = queries.Length / index.Dimensions;
        var distances = new float[count * K];
        var labels = new long[count * K];
        index.SearchWithParams(count, queries, K, parameters, distances, labels);
        return (distances, labels);
    }
}
