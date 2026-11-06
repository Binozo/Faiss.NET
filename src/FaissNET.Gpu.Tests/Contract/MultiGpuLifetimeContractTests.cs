using System.Runtime.CompilerServices;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class MultiGpuLifetimeContractTests
{
    private const int Stored = 64;

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void Squad_OutlivesItsProvidersAndCpuSource(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var reference = c.Single.CreatePopulatedCpuIndex(stored);
        var cpu = c.Single.CreatePopulatedCpuIndex(stored);
        var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        cpu.Dispose();
        providers.Dispose();

        SearchAssert.SameResults(reference, squad, queries, 5, relTol: c.Single.DistanceTolerance, absTol: 1e-3f);
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void DisposedSquad_EveryMemberThrowsObjectDisposed(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        var squad = c.Transfer(mode, providers, [0, 0], cpu);

        squad.Dispose();

        Assert.Throws<ObjectDisposedException>(() => squad.TotalCount);
        Assert.Throws<ObjectDisposedException>(() => SearchAssert.Search(squad, new float[c.Dimensions], 1));
        Assert.Throws<ObjectDisposedException>(() => ((ITrainableFloatIndex)squad).IsTrained);
        Assert.Throws<ObjectDisposedException>(() => c.DeviceIds(squad));
        Assert.Throws<ObjectDisposedException>(() => c.MemberCounts(squad));
        Assert.Throws<ObjectDisposedException>(() => c.ToCpu(squad));
    }

    [Fact]
    public void DisposedSquads_RefuseNewMembers()
    {
        using var cpu = new IndexFlatL2(4);
        using var providers = new ProviderSet(3);
        var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items[..2], [0, 0], cpu);
        var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items[..2], [0, 0], cpu);
        using var member = GpuIndexProvider.TransferToGpu(providers.Items[2], cpu);

        shards.Dispose();
        replicas.Dispose();

        Assert.Throws<ObjectDisposedException>(() => shards.AddIndex(member));
        Assert.Throws<ObjectDisposedException>(() => replicas.AddReplica(member));
        Assert.Equal(0, member.TotalCount);
    }

    [Fact]
    public void ReplicaViews_AreTypedCachedAndDieWithTheirParent()
    {
        var last = FaissGpu.GetNumGpus() - 1;
        var c = GpuIndexCatalog.Get("IVFFlatL2");
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, last], (IndexIVFFlat<IndexFlatL2>)cpu);

        var view = Assert.IsType<GpuIndexIVFFlat<IndexFlatL2>>(replicas[1]);
        Assert.Same(view, replicas[1]);
        Assert.Equal(last, view.DeviceId);
        Assert.Equal(8, view.TotalCount);

        replicas.Dispose();

        Assert.Throws<ObjectDisposedException>(() => view.TotalCount);
    }

    [Fact]
    public void Finalization_OfAbandonedSquadsViewsAndProviders_IsSafe()
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: 8);
        var queries = Vectors.Random(seed: 2, count: 4, dimensions: 8);

        for (var i = 0; i < 8; i++)
            Abandon(stored);

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        using var cpu = new IndexFlatL2(8);
        cpu.Add(Stored, stored);
        using var providers = new ProviderSet(2);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu);
        var (distances, labels) = SearchAssert.Search(shards, queries, 3);
        SearchAssert.MatchesBruteForce(Faiss.Models.MetricType.L2, stored, queries, 3, distances, labels);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Abandon(float[] stored)
    {
        var cpu = new IndexFlatL2(8);
        cpu.Add(Stored, stored);

        // Undisposed providers keep their temporary device memory (1.5 GiB by default) until finalized.
        var providers = new[] { new GpuResourcesProvider(), new GpuResourcesProvider() };
        foreach (var provider in providers)
            provider.SetTempMemory(16 * 1024 * 1024);

        var shards = GpuIndexProvider.TransferToGpuSharded(providers, [0, 0], cpu);
        var replicas = GpuIndexProvider.TransferToGpuReplicated(providers, [0, 0], cpu);
        _ = shards[1].TotalCount + replicas[1].TotalCount;
        _ = shards.DeviceIds.Length + replicas.DeviceIds.Length;
    }
}
