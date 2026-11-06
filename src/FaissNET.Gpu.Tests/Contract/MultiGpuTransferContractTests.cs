using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class MultiGpuTransferContractTests
{
    private const int Stored = 64;
    private const int QueryCount = 12;
    private const int K = 5;

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.AllOnEveryDeviceSet), MemberType = typeof(MultiGpuCatalog))]
    public void Transfer_SpreadsVectorsByMode(MultiGpuIndexCase c, MultiGpuMode mode, int[] devices)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(devices.Length);

        using var squad = c.Transfer(mode, providers, devices, cpu);

        Assert.Equal(c.ExpectedMemberCounts(mode, Stored, devices.Length), c.MemberCounts(squad));
        Assert.Equal(Stored, squad.TotalCount);
        Assert.Equal(c.Dimensions, squad.Dimensions);
        Assert.Equal(c.Single.Metric, squad.Metric);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.ExactOnEveryDeviceSet), MemberType = typeof(MultiGpuCatalog))]
    public void Search_MatchesBruteForce(MultiGpuIndexCase c, MultiGpuMode mode, int[] devices)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(stored);
        using var providers = new ProviderSet(devices.Length);
        using var squad = c.Transfer(mode, providers, devices, cpu);

        var (distances, labels) = SearchAssert.Search(squad, queries, K);

        // Labels are source positions: flat shards shift each shard's labels by the vectors before it, IVF shards keep the ids.
        SearchAssert.MatchesBruteForce(c.Single.Metric, stored, queries, K, distances, labels);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.Approximate), MemberType = typeof(MultiGpuCatalog))]
    public void Search_MatchesCpuSource(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        SearchAssert.SameResults(cpu, squad, queries, K, relTol: c.Single.DistanceTolerance, absTol: 1e-3f);
    }

    /// <summary>A stored vector held by the second shard comes back under its source position, not its position in that shard.</summary>
    [Theory]
    [InlineData("FlatL2")]
    [InlineData("FlatGenericL2")]
    [InlineData("IVFFlatL2")]
    public void Sharded_StoredVectorInLastShard_IsFoundUnderItsSourcePosition(string name)
    {
        var c = MultiGpuCatalog.Get(name);
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(stored);
        using var providers = new ProviderSet(2);
        using var shards = c.Transfer(MultiGpuMode.Sharded, providers, [0, 0], cpu);

        var (distances, labels) = SearchAssert.Search(shards, stored[(51 * c.Dimensions)..(52 * c.Dimensions)], 1);

        Assert.Equal(51, labels[0]);
        FloatAssert.Equal(0f, distances[0], relTol: 0, absTol: 1e-4f);
    }

    /// <summary>Missing neighbours stay -1: the shards' label shift must skip them.</summary>
    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Search_KExceedsTotalCount_PadsWithUnshiftedSentinels(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        const int stored = 3;
        const int k = 6;
        var vectors = Vectors.Random(seed: 1, count: stored, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(vectors);
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        var (_, labels) = SearchAssert.Search(squad, vectors[..c.Dimensions], k);

        Assert.Equal([0L, 1L, 2L], labels[..stored].Order());
        Assert.All(labels[stored..], label => Assert.Equal(-1, label));
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Transfer_EmptySource_SearchFindsNothing(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreateCpuIndex();
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        var (_, labels) = SearchAssert.Search(squad, Vectors.Random(seed: 2, count: 2, dimensions: c.Dimensions), K);

        Assert.Equal([0L, 0L], c.MemberCounts(squad));
        Assert.All(labels, label => Assert.Equal(-1, label));
    }

    /// <summary>Five members for three vectors: some shards stay empty and search still finds every vector.</summary>
    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Transfer_MoreDevicesThanVectors_LeavesShardsEmptyAndFindsEverything(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        const int stored = 3;
        int[] devices = [0, 0, 0, 0, 0];
        var vectors = Vectors.Random(seed: 1, count: stored, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(vectors);
        using var providers = new ProviderSet(devices.Length);
        using var squad = c.Transfer(mode, providers, devices, cpu);

        var (_, labels) = SearchAssert.Search(squad, vectors, stored);

        Assert.Equal(c.ExpectedMemberCounts(mode, stored, devices.Length), c.MemberCounts(squad));
        for (var q = 0; q < stored; q++)
            Assert.Equal([0L, 1L, 2L], labels[(q * stored)..((q + 1) * stored)].Order());
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Transfer_LeavesSourceUnchanged(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(stored);
        using var reference = c.Single.CreatePopulatedCpuIndex(stored);
        using var providers = new ProviderSet(2);

        using (c.Transfer(mode, providers, [0, 0], cpu))
        {
        }

        Assert.Equal(Stored, cpu.TotalCount);
        SearchAssert.SameResults(reference, cpu, queries, K, relTol: 0, absTol: 0);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Transfer_Twice_GivesIndependentIndexes(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var first = c.Transfer(mode, providers, [0, 0], cpu);
        using var second = c.Transfer(mode, providers, [0, 0], cpu);

        first.Reset();

        Assert.Equal(0, first.TotalCount);
        Assert.Equal(Stored, second.TotalCount);
        Assert.Equal(c.ExpectedMemberCounts(mode, Stored, 2), c.MemberCounts(second));
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.AllOnEveryDeviceSet), MemberType = typeof(MultiGpuCatalog))]
    public void DeviceIds_FollowTheRequestedOrder(MultiGpuIndexCase c, MultiGpuMode mode, int[] devices)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(devices.Length);
        using var squad = c.Transfer(mode, providers, devices, cpu);

        Assert.Equal(devices, c.DeviceIds(squad));
    }
}
