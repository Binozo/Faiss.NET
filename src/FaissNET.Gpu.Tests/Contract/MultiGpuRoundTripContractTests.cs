using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class MultiGpuRoundTripContractTests
{
    private const int Stored = 64;
    private const int K = 5;

    public static TheoryData<MultiGpuIndexCase> Ivf { get; } = IvfCases();

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void ToCpu_HoldsTheSourceOnceAndSearchesLikeTheDeviceIndex(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        var queries = Vectors.Random(seed: 2, count: 12, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);

        using var back = c.ToCpu(squad);

        // Shards merge every part
        // Replicas give back one replica
        Assert.Equal(Stored, back.TotalCount);
        SearchAssert.SameResults(squad, back, queries, K, relTol: c.Single.DistanceTolerance, absTol: 1e-3f);
        Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), SearchAssert.StoredLabels(back, Stored));
    }

    [Theory]
    [MemberData(nameof(MultiGpuCatalog.All), MemberType = typeof(MultiGpuCatalog))]
    public void ToCpu_ResultIsIndependentOfTheDeviceIndex(MultiGpuIndexCase c, MultiGpuMode mode)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);
        using var back = c.ToCpu(squad);

        ((IIDSequentialFloatIndex)back).Add(1, Vectors.Random(seed: 3, count: 1, dimensions: c.Dimensions));

        Assert.Equal(Stored + 1, back.TotalCount);
        Assert.Equal(Stored, squad.TotalCount);
    }

    /// <summary>Ivf storage keeps no ids on the device, so the replicas cannot be copied back.</summary>
    [Theory]
    [MemberData(nameof(Ivf))]
    public void Replicated_IvfStorage_ToCpu_ThrowsNative(MultiGpuIndexCase c)
    {
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Ivf };
        using var replicas = c.ToReplicated(providers.Items, [0, 0], cpu, options);

        Assert.Throws<FaissNativeException>(() => c.ToCpu(replicas).Dispose());
        Assert.Equal(8, replicas.TotalCount);
    }

    private static TheoryData<MultiGpuIndexCase> IvfCases()
    {
        var data = new TheoryData<MultiGpuIndexCase>();
        foreach (var name in new[] { "IVFFlatL2", "IVFSQ8", "IVFPQ" })
            data.Add(MultiGpuCatalog.Get(name));

        return data;
    }
}
