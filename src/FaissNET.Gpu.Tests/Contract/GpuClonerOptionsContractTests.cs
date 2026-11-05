using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class GpuClonerOptionsContractTests
{
    private const int Stored = 64;
    private const int K = 5;

    public static TheoryData<string> Flags { get; } = new(Flag.All.Keys);

    [Fact]
    public void New_MatchesFaissDefaults()
    {
        using var options = new GpuClonerOptions();

        Assert.Equal(IndicesOptions.Bit64, options.StorageMode);
        Assert.Equal(0, options.ReserveVecs);
        Assert.All(Flag.All.Values, flag => Assert.False(flag.Get(options)));
    }

    [Fact]
    public void NewMultiple_MatchesFaissDefaults()
    {
        using var options = new GpuMultipleClonerOptions();

        Assert.False(options.Shard);
        Assert.Equal(GpuShardType.IdModulo, options.ShardType);
        Assert.Equal(IndicesOptions.Bit64, options.StorageMode);
        Assert.All(Flag.All.Values, flag => Assert.False(flag.Get(options)));
    }

    [Theory]
    [MemberData(nameof(Flags))]
    public void Flag_SetAlone_ChangesOnlyItself(string name)
    {
        using var options = new GpuClonerOptions();

        Flag.All[name].Set(options, true);

        Assert.All(Flag.All, flag => Assert.Equal(flag.Key == name, flag.Value.Get(options)));

        Flag.All[name].Set(options, false);
        Assert.False(Flag.All[name].Get(options));
    }

    [Theory]
    [MemberData(nameof(Flags))]
    public void Flag_OnMultipleOptions_ChangesOnlyItself(string name)
    {
        using var options = new GpuMultipleClonerOptions();

        Flag.All[name].Set(options, true);

        Assert.All(Flag.All, flag => Assert.Equal(flag.Key == name, flag.Value.Get(options)));
        Assert.False(options.Shard);
    }

    [Theory]
    [InlineData(IndicesOptions.Cpu)]
    [InlineData(IndicesOptions.Ivf)]
    [InlineData(IndicesOptions.Bit32)]
    [InlineData(IndicesOptions.Bit64)]
    public void StorageMode_RoundTrips(IndicesOptions mode)
    {
        using var options = new GpuClonerOptions { StorageMode = mode };

        Assert.Equal(mode, options.StorageMode);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(4)]
    public void StorageMode_UndefinedValue_ThrowsAndKeepsPrevious(int value)
    {
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Bit32 };

        Assert.Throws<ArgumentOutOfRangeException>(() => options.StorageMode = (IndicesOptions)value);

        Assert.Equal(IndicesOptions.Bit32, options.StorageMode);
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(4096L)]
    [InlineData(5_000_000_000L)]
    public void ReserveVecs_RoundTripsBeyond32Bits(long value)
    {
        using var options = new GpuClonerOptions { ReserveVecs = value };

        Assert.Equal(value, options.ReserveVecs);
    }

    [Theory]
    [InlineData(GpuShardType.IdModulo)]
    [InlineData(GpuShardType.IdRange)]
    [InlineData(GpuShardType.InvertedList)]
    public void ShardType_RoundTrips(GpuShardType type)
    {
        using var options = new GpuMultipleClonerOptions { ShardType = type };

        Assert.Equal(type, options.ShardType);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(5)]
    public void ShardType_UndefinedValue_ThrowsAndKeepsPrevious(int value)
    {
        using var options = new GpuMultipleClonerOptions { ShardType = GpuShardType.IdRange };

        Assert.Throws<ArgumentOutOfRangeException>(() => options.ShardType = (GpuShardType)value);

        Assert.Equal(GpuShardType.IdRange, options.ShardType);
    }

    [Fact]
    public void Disposed_ThrowsObjectDisposed()
    {
        var options = new GpuClonerOptions();
        options.Dispose();

        Assert.Throws<ObjectDisposedException>(() => options.UseFloat16);
        Assert.Throws<ObjectDisposedException>(() => options.StorageMode = IndicesOptions.Bit32);
        Assert.Throws<ObjectDisposedException>(() => options.ReserveVecs);
    }

    [Fact]
    public void Disposed_TransferWithOptions_ThrowsObjectDisposed()
    {
        using var cpu = new IndexFlatL2(4);
        using var provider = new GpuResourcesProvider();
        var options = new GpuClonerOptions();
        options.Dispose();

        Assert.Throws<ObjectDisposedException>(() => GpuIndexProvider.TransferToGpu(provider, cpu, options));
    }

    [Fact]
    public void DisposeTwice_IsHarmless()
    {
        var options = new GpuMultipleClonerOptions();
        options.Dispose();
        options.Dispose();

        Assert.Throws<ObjectDisposedException>(() => options.Shard);
    }

    [Fact]
    public void Transfer_NullOptions_ThrowsArgumentNull()
    {
        using var cpu = new IndexFlatL2(4);
        using var provider = new GpuResourcesProvider();

        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu(provider, cpu, (GpuClonerOptions)null!));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void DefaultOptions_BehaveLikeTransferWithoutOptions(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        using var plain = c.ToGpu(provider, cpu, 0);
        using var withOptions = c.ToGpuWithOptions(provider, cpu, options, 0);

        SearchAssert.SameResults(plain, withOptions, queries, K, relTol: 0, absTol: 0);
    }

    [Theory]
    [MemberData(nameof(IvfStorageModes))]
    public void StorageMode_KeepingIds_ReportsCallerIds(string caseName, IndicesOptions mode)
    {
        var c = GpuIndexCatalog.Get(caseName);
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var ids = Enumerable.Range(0, Stored).Select(i => 1_000_000L + 7L * i).ToArray();

        using var cpu = c.CreateCpuIndex();
        ((IIDMappedFloatIndex)cpu).Add(Stored, stored, ids);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = mode };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        Assert.Equal(ids, SearchAssert.StoredLabels(gpu, Stored));
        SearchAssert.SameResults(cpu, gpu, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    /// <summary>
    /// <see cref="IndicesOptions.Ivf"/> keeps no ids: search reports (inverted list &lt;&lt; 32 | offset in list) instead and such an index cannot come back to the CPU.
    /// </summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Ivf), MemberType = typeof(GpuIndexCatalog))]
    public void StorageMode_Ivf_ReportsListAndOffset(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        var nlist = ((IIVFIndex)cpu).Nlist;
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Ivf };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        // Every list is probed, so one query with k = everything reports each stored vector exactly once.
        var decoded = SearchAssert.StoredLabels(gpu, Stored).Select(label => (List: label >> 32, Offset: label & uint.MaxValue)).ToArray();

        Assert.All(decoded, d => Assert.InRange(d.List, 0, nlist - 1));
        Assert.Equal(Stored, decoded.Distinct().Count());
        Assert.All(
            decoded.GroupBy(d => d.List),
            list => Assert.Equal(Enumerable.Range(0, list.Count()).Select(i => (long)i), list.Select(d => d.Offset).Order()));

        Assert.Throws<FaissNativeException>(() => c.ToCpu(gpu).Dispose());
    }

    /// <summary>
    /// Every IVF shard reports (list, offset) labels for its own part of the same lists and faiss merges them untranslated,
    /// so labels from different shards collide. Replicas hold identical lists, so their labels stay meaningful.
    /// </summary>
    [Fact]
    public async Task StorageMode_Ivf_ShardedIvf_ThrowsAndReplicatedKeepsListAndOffset()
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: 8);
        using var cpu = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4, ownQuantizer: true) { Nprobe = 4 };
        await cpu.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8));
        cpu.Add(Stored, stored);
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Ivf };

        var sharded = Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu, options));
        Assert.Equal("options", sharded.ParamName);

        using var replicas = GpuIndexProvider.TransferToGpuReplicated([first, second], [0, 0], cpu, options);
        var decoded = SearchAssert.StoredLabels(replicas, Stored).Select(label => (List: label >> 32, Offset: label & uint.MaxValue)).ToArray();
        Assert.Equal(Stored, decoded.Distinct().Count());
        Assert.All(decoded, d => Assert.InRange(d.List, 0, 3));
    }

    /// <summary>A flat index has no inverted lists and ignores <see cref="GpuClonerOptions.StorageMode"/>.</summary>
    [Fact]
    public void StorageMode_Ivf_ShardedFlat_Transfers()
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: 8);
        using var cpu = new IndexFlatL2(8);
        cpu.Add(Stored, stored);
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();
        using var options = new GpuMultipleClonerOptions { StorageMode = IndicesOptions.Ivf };

        using var shards = GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu, options);

        Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), SearchAssert.StoredLabels(shards, Stored));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Ivf), MemberType = typeof(GpuIndexCatalog))]
    public void StorageMode_Bit32_SourceIdBeyond32Bits_ThrowsInsteadOfAborting(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        using var cpu = c.CreateCpuIndex();
        ((IIDMappedFloatIndex)cpu).Add(4, stored, [1, 2, (long)int.MaxValue + 1, 3]);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Bit32 };

        Assert.Throws<ArgumentException>(() => c.ToGpuWithOptions(provider, cpu, options, 0));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Ivf), MemberType = typeof(GpuIndexCatalog))]
    public void StorageMode_Bit32_SourceIdsAtTheLimits_Transfer(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);
        long[] ids = [int.MinValue, int.MaxValue];
        using var cpu = c.CreateCpuIndex();
        ((IIDMappedFloatIndex)cpu).Add(2, stored, ids);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Bit32 };

        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        Assert.Equal(ids, SearchAssert.StoredLabels(gpu, 2));
    }

    /// <summary>Faiss truncates ids added on a 32-bit device index without checking, so the wrapper must reject them.</summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void StorageMode_Bit32_DeviceAddBeyond32Bits_ThrowsAndAddsNothing(GpuIndexCase c)
    {
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Bit32 };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);

        Assert.Throws<ArgumentOutOfRangeException>(() => ((IIDMappedFloatIndex)gpu).Add(2, vectors, [5, 1L << 33]));
        Assert.Throws<ArgumentOutOfRangeException>(() => ((IIDMappedFloatIndex)gpu).Add(2, vectors, [5, (long)int.MinValue - 1]));

        Assert.Equal(0, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void StorageMode_Bit64_DeviceAddBeyond32Bits_KeepsIds(GpuIndexCase c)
    {
        long[] ids = [-(1L << 35), 1L << 40];
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { StorageMode = IndicesOptions.Bit64 };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);

        ((IIDMappedFloatIndex)gpu).Add(2, vectors, ids);

        Assert.Equal(ids, SearchAssert.StoredLabels(gpu, 2));
    }

    /// <summary>Float16 flat storage rounds every component to the nearest half, observable through reconstruction.</summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceReconstructable), MemberType = typeof(GpuIndexCatalog))]
    public void UseFloat16_OnFlat_StoresHalfPrecision(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { UseFloat16 = true };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        var reconstructed = ((IReconstructFloatIndex)gpu).Reconstruct(0, 8);

        Assert.Equal(stored.Select(x => (float)(Half)x), reconstructed);
        Assert.NotEqual(stored, reconstructed);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceReconstructable), MemberType = typeof(GpuIndexCatalog))]
    public void UseFloat16Off_OnFlat_StoresFullPrecision(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { UseFloat16 = false };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        Assert.Equal(stored, ((IReconstructFloatIndex)gpu).Reconstruct(0, 8));
    }

    /// <summary>The flags below change the device layout or arithmetic, never which neighbours an exact index finds.</summary>
    [Theory]
    [MemberData(nameof(ExactLayoutFlags))]
    public void LayoutFlag_KeepsExactSearchExact(string caseName, string flag)
    {
        var c = GpuIndexCatalog.Get(caseName);
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();
        Flag.All[flag].Set(options, true);
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        var (distances, labels) = SearchAssert.Search(gpu, queries, K);

        SearchAssert.MatchesBruteForce(c.Metric, stored, queries, K, distances, labels);
    }

    /// <summary>Precomputed tables and float16 lookup tables only change how IVFPQ distances are computed.</summary>
    [Theory]
    [InlineData(nameof(GpuClonerOptions.UsePrecomputed))]
    [InlineData(nameof(GpuClonerOptions.UseFloat16))]
    [InlineData(nameof(GpuClonerOptions.UseFloat16CoarseQuantizer))]
    public void IvfPqComputeFlag_MatchesCpuSource(string flag)
    {
        var c = GpuIndexCatalog.Get("IVFPQ");
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();
        Flag.All[flag].Set(options, true);
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        SearchAssert.SameResults(cpu, gpu, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: 2e-3f, absTol: 2e-3f);
    }

    /// <summary>Reserving list memory on an empty IVF index must not change what it holds once vectors arrive.</summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void ReserveVecs_OnEmptyIvf_DeviceAddStillMatchesCpu(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions { ReserveVecs = 4 * Stored };
        using var gpu = c.ToGpuWithOptions(provider, cpu, options, 0);

        ((IIDSequentialFloatIndex)gpu).Add(Stored, stored);
        ((IIDSequentialFloatIndex)cpu).Add(Stored, stored);

        Assert.Equal(Stored, gpu.TotalCount);
        SearchAssert.SameResults(cpu, gpu, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    public static TheoryData<string, IndicesOptions> IvfStorageModes()
    {
        var data = new TheoryData<string, IndicesOptions>();
        foreach (var c in GpuIndexCatalog.With(GpuCapabilities.Ivf))
        {
            foreach (var mode in new[] { IndicesOptions.Cpu, IndicesOptions.Bit32, IndicesOptions.Bit64 })
            {
                data.Add(c.Name, mode);
            }
        }

        return data;
    }

    /// <summary>Transposed storage applies to device flats, float16 coarse quantization to device IVF.</summary>
    public static TheoryData<string, string> ExactLayoutFlags()
    {
        var data = new TheoryData<string, string>();
        foreach (var c in GpuIndexCatalog.With(GpuCapabilities.Exact))
        {
            data.Add(c.Name, c.Has(GpuCapabilities.Ivf)
                ? nameof(GpuClonerOptions.UseFloat16CoarseQuantizer)
                : nameof(GpuClonerOptions.StoreTransposed));
        }

        return data;
    }

    private sealed record Flag(Func<GpuClonerOptions, bool> Get, Action<GpuClonerOptions, bool> Set)
    {
        public static readonly IReadOnlyDictionary<string, Flag> All = new Dictionary<string, Flag>
        {
            [nameof(GpuClonerOptions.UseFloat16)] = new(o => o.UseFloat16, (o, v) => o.UseFloat16 = v),
            [nameof(GpuClonerOptions.UseFloat16CoarseQuantizer)] = new(o => o.UseFloat16CoarseQuantizer, (o, v) => o.UseFloat16CoarseQuantizer = v),
            [nameof(GpuClonerOptions.UsePrecomputed)] = new(o => o.UsePrecomputed, (o, v) => o.UsePrecomputed = v),
            [nameof(GpuClonerOptions.StoreTransposed)] = new(o => o.StoreTransposed, (o, v) => o.StoreTransposed = v),
            [nameof(GpuClonerOptions.Verbose)] = new(o => o.Verbose, (o, v) => o.Verbose = v),
        };
    }
}
