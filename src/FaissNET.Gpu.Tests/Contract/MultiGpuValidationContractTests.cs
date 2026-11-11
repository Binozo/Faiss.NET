using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class MultiGpuValidationContractTests
{
    private const int Stored = 64;

    public static TheoryData<MultiGpuMode> Modes { get; } = new(MultiGpuMode.Sharded, MultiGpuMode.Replicated);

    public static TheoryData<int> InvalidDevices { get; } = new(-1, FaissGpu.GetNumGpus(), int.MaxValue);

    [Theory]
    [MemberData(nameof(Modes))]
    public void NullArguments_ThrowArgumentNull(MultiGpuMode mode)
    {
        using var cpu = Flat();
        using var providers = new ProviderSet(2);

        Assert.Equal("contexts", Assert.Throws<ArgumentNullException>(() => Transfer(mode, null!, [0, 0], cpu)).ParamName);
        Assert.Equal("deviceIds", Assert.Throws<ArgumentNullException>(() => Transfer(mode, providers.Items, null!, cpu)).ParamName);
        Assert.Equal("cpuIndex", Assert.Throws<ArgumentNullException>(() => Transfer(mode, providers.Items, [0, 0], (IndexFlatL2)null!)).ParamName);
        Assert.Equal("contexts", Assert.Throws<ArgumentNullException>(() => Transfer(mode, [providers.Items[0], null!], [0, 0], cpu)).ParamName);
        Assert.Equal("contexts", Assert.Throws<ArgumentNullException>(() => Transfer(mode, [null!, null!], [0, 0], cpu)).ParamName);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void EmptyOrMismatchedArrays_ThrowArgument(MultiGpuMode mode)
    {
        using var cpu = Flat();
        using var providers = new ProviderSet(3);

        Assert.Equal("contexts", Assert.Throws<ArgumentException>(() => Transfer(mode, [], [], cpu)).ParamName);
        Assert.Equal("contexts", Assert.Throws<ArgumentException>(() => Transfer(mode, providers.Items[..2], [0, 0, 0], cpu)).ParamName);
        Assert.Equal("contexts", Assert.Throws<ArgumentException>(() => Transfer(mode, providers.Items, [0, 0], cpu)).ParamName);
    }

    /// <summary>With one device faiss returns a plain device index instead of a shards or replicas index.</summary>
    [Theory]
    [MemberData(nameof(Modes))]
    public void SingleDevice_ThrowsArgument(MultiGpuMode mode)
    {
        using var cpu = Flat();
        using var providers = new ProviderSet(1);

        Assert.Equal("deviceIds", Assert.Throws<ArgumentException>(() => Transfer(mode, providers.Items, [0], cpu)).ParamName);
    }

    [Theory]
    [MemberData(nameof(InvalidDevices))]
    public void InvalidDevice_ThrowsArgumentOutOfRangeAndLeavesProvidersUsable(int device)
    {
        using var cpu = Flat();
        using var providers = new ProviderSet(2);

        foreach (var mode in Enum.GetValues<MultiGpuMode>())
            Assert.Equal("deviceIds", Assert.Throws<ArgumentOutOfRangeException>(() => Transfer(mode, providers.Items, [0, device], cpu)).ParamName);

        using var gpu = GpuIndexProvider.TransferToGpu(providers.Items[0], cpu);
        Assert.Equal(Stored, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void DisposedProvider_ThrowsObjectDisposedAndLeavesTheOtherUsable(MultiGpuMode mode)
    {
        using var cpu = Flat();
        using var live = new GpuResourcesProvider();
        var disposed = new GpuResourcesProvider();
        disposed.Dispose();

        Assert.Throws<ObjectDisposedException>(() => Transfer(mode, [live, disposed], [0, 0], cpu));

        using var gpu = GpuIndexProvider.TransferToGpu(live, cpu);
        Assert.Equal(Stored, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void DisposedSourceOrOptions_ThrowObjectDisposed(MultiGpuMode mode)
    {
        var disposedSource = Flat();
        disposedSource.Dispose();
        using var cpu = Flat();
        using var providers = new ProviderSet(2);
        var disposedOptions = new GpuMultipleClonerOptions();
        disposedOptions.Dispose();

        Assert.Throws<ObjectDisposedException>(() => Transfer(mode, providers.Items, [0, 0], disposedSource));
        Assert.Throws<ObjectDisposedException>(() => Transfer(mode, providers.Items, [0, 0], cpu, disposedOptions));
    }

    /// <summary>Without options faiss's default is to replicate. The sharded transfer must still shard.</summary>
    [Fact]
    public void Sharded_WithoutOptions_SplitsTheVectors()
    {
        using var cpu = Ivf();
        using var providers = new ProviderSet(2);

        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu);

        Assert.Equal([Stored / 2L, Stored / 2L], Counts(shards));
    }

    [Fact]
    public void SameOptions_ReusedAcrossModes_GiveTheRequestedModeEachTime()
    {
        using var cpu = Ivf();
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions();

        using var firstShards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu, options);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu, options);
        using var secondShards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu, options);

        Assert.Equal([Stored / 2L, Stored / 2L], Counts(firstShards));
        Assert.Equal([(long)Stored, Stored], [replicas[0].TotalCount, replicas[1].TotalCount]);
        Assert.Equal([Stored / 2L, Stored / 2L], Counts(secondShards));
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void Bit32_SourceIdBeyond32Bits_ThrowsArgumentAndKeepsSource(MultiGpuMode mode)
    {
        const long wide = int.MaxValue + 1L;
        using var cpu = Ivf(ids: [wide]);
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Bit32 };

        Assert.Equal("cpuIndex", Assert.Throws<ArgumentException>(() => Transfer(mode, providers.Items, [0, 0], cpu, options)).ParamName);
        Assert.Equal([wide], SearchAssert.StoredLabels(cpu, 1));
    }

    [Fact]
    public void Bit32_ShardsAddIdBeyond32Bits_ThrowsAndAddsNothing()
    {
        using var cpu = Ivf();
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Bit32 };
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu, options);
        var vector = Vectors.Random(seed: 7, count: 1, dimensions: 8);

        Assert.Equal("xids", Assert.Throws<ArgumentOutOfRangeException>(() => shards.Add(1, vector, [int.MaxValue + 1L])).ParamName);
        Assert.Equal(Stored, shards.TotalCount);

        shards.Add(1, vector, [int.MaxValue]);
        Assert.Equal(Stored + 1, shards.TotalCount);
        var (_, labels) = SearchAssert.Search(shards, vector, 1);
        Assert.Equal(int.MaxValue, labels[0]);
    }

    [Fact]
    public void Bit32_ReplicasAddIdBeyond32Bits_ThrowsAndAddsNothing()
    {
        using var cpu = Ivf();
        using var providers = new ProviderSet(2);
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Bit32 };
        using var replicas = GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu, options);
        var vector = Vectors.Random(seed: 7, count: 1, dimensions: 8);

        Assert.Equal("xids", Assert.Throws<ArgumentOutOfRangeException>(() => replicas.Add(1, vector, [int.MaxValue + 1L])).ParamName);
        Assert.Equal(Stored, replicas.TotalCount);

        replicas.Add(1, vector, [int.MaxValue]);
        Assert.Equal(Stored + 1, replicas.TotalCount);
        var (_, labels) = SearchAssert.Search(replicas, vector, 1);
        Assert.Equal(int.MaxValue, labels[0]);
    }

    /// <summary>Device IVFPQ needs 8-bit codes and a supported number of sub-quantizers. A sharded transfer would abort on either.</summary>
    [Theory]
    [InlineData(16, 4, 4)]
    [InlineData(20, 5, 8)]
    public async Task IvfPq_UnsupportedCodes_ThrowCloningUnsupportedOnEveryPathAndKeepSource(int dimensions, int m, int bits)
    {
        using var cpu = new IndexIVFPQ(dimensions, n: 4, m: m, subQuantizer: bits) { Nprobe = 4 };
        await cpu.TrainAsync(1024, Vectors.Random(seed: 98, count: 1024, dimensions: dimensions));
        cpu.Add(8, Vectors.Random(seed: 1, count: 8, dimensions: dimensions));
        using var providers = new ProviderSet(2);

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(providers.Items[0], cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu));
        Assert.Equal(8, cpu.TotalCount);
    }

    /// <summary>64 sub-quantizers need 64 KiB of float32 lookup tables, beyond 48 KiB of shared memory (float16 tables fit).</summary>
    [Fact]
    public async Task IvfPq_64SubQuantizers_ShardOnlyWithFloat16LookupTables()
    {
        const int dimensions = 64;
        using var cpu = new IndexIVFPQ(dimensions, n: 4, m: 64) { Nprobe = 4 };
        await cpu.TrainAsync(1024, Vectors.Random(seed: 98, count: 1024, dimensions: dimensions));
        cpu.Add(8, Vectors.Random(seed: 1, count: 8, dimensions: dimensions));
        using var providers = new ProviderSet(2);
        using var float16 = new GpuMultipleClonerOptions { UseFloat16 = true };

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu));

        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu, float16);
        Assert.Equal(8, shards.TotalCount);
        Assert.Equal(2, shards.Count);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2049)]
    public void Nprobe_OutsideDeviceRange_ThrowsCloningUnsupportedWhenSharded(int nprobe)
    {
        using var cpu = Ivf();
        cpu.Nprobe = nprobe;
        using var providers = new ProviderSet(2);

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu));
        Assert.Equal(nprobe, cpu.Nprobe);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2048)]
    public void Nprobe_AtDeviceRangeLimits_Shards(int nprobe)
    {
        using var cpu = Ivf();
        cpu.Nprobe = nprobe;
        using var providers = new ProviderSet(2);

        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu);

        Assert.Equal(Stored, shards.TotalCount);
    }

    [Theory]
    [InlineData(MetricType.L1)]
    [InlineData(MetricType.Linf)]
    public void Flat_NonL2OrIpMetric_ThrowsCloningUnsupportedOnEveryMultiPath(MetricType metric)
    {
        using var cpu = new IndexFlat(4, metric);
        using var providers = new ProviderSet(2);

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuReplicated(providers.Items, [0, 0], cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0], cpu));
    }

    private static IndexFlatL2 Flat()
    {
        var index = new IndexFlatL2(8);
        index.Add(Stored, Vectors.Random(seed: 1, count: Stored, dimensions: 8));
        return index;
    }

    /// <summary>An exhaustive IVFFlatL2 holding <see cref="Stored"/> vectors with ids 0..63, or the given ids.</summary>
    private static IndexIVFFlat<IndexFlatL2> Ivf(long[]? ids = null)
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4, ownQuantizer: true) { Nprobe = 4 };
        index.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8)).GetAwaiter().GetResult();
        ids ??= Enumerable.Range(0, Stored).Select(i => (long)i).ToArray();
        index.Add(ids.Length, Vectors.Random(seed: 1, count: ids.Length, dimensions: 8), ids);
        return index;
    }

    private static long[] Counts(GpuIndexShards<GpuIndexIVFFlat<IndexFlatL2>, IndexIVFFlat<IndexFlatL2>> shards) =>
        Enumerable.Range(0, shards.Count).Select(i => shards[i].TotalCount).ToArray();

    private static IDisposable Transfer(MultiGpuMode mode, GpuResourcesProvider[] providers, int[] devices, IndexFlatL2 cpu, GpuMultipleClonerOptions? options = null) =>
        mode == MultiGpuMode.Sharded
            ? GpuIndexProvider.TransferToGpuSharded(providers, devices, cpu, options)
            : GpuIndexProvider.TransferToGpuReplicated(providers, devices, cpu, options);

    private static IDisposable Transfer(MultiGpuMode mode, GpuResourcesProvider[] providers, int[] devices, IndexIVFFlat<IndexFlatL2> cpu, GpuMultipleClonerOptions? options = null) =>
        mode == MultiGpuMode.Sharded
            ? GpuIndexProvider.TransferToGpuSharded(providers, devices, cpu, options)
            : GpuIndexProvider.TransferToGpuReplicated(providers, devices, cpu, options);
}
