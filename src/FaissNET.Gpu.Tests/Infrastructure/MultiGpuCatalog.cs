using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Resources;
using Xunit;

namespace Faiss.Gpu.Tests.Infrastructure;

public enum MultiGpuMode
{
    Sharded,
    Replicated,
}

public sealed record MultiGpuIndexCase(
    GpuIndexCase Single,
    Func<GpuResourcesProvider[], int[], FloatIndex, GpuMultipleClonerOptions?, FloatIndex> ToSharded,
    Func<GpuResourcesProvider[], int[], FloatIndex, GpuMultipleClonerOptions?, FloatIndex> ToReplicated,
    Func<FloatIndex, FloatIndex> ToCpu,
    Func<FloatIndex, long[]> MemberCounts,
    Func<FloatIndex, int[]> DeviceIds)
{
    public string Name => Single.Name;

    public int Dimensions => Single.Dimensions;

    /// <summary>Flat shards hold contiguous blocks with successive ids. IVF shards are split by <see cref="GpuShardType"/>.</summary>
    public bool IsFlat => !Single.Has(GpuCapabilities.Ivf);

    public FloatIndex Transfer(MultiGpuMode mode, ProviderSet providers, int[] devices, FloatIndex cpu, GpuMultipleClonerOptions? options = null) =>
        mode == MultiGpuMode.Sharded
            ? ToSharded(providers.Items, devices, cpu, options)
            : ToReplicated(providers.Items, devices, cpu, options);

    /// <summary>
    /// Vectors per member right after transferring <paramref name="stored"/> vectors with ids 0..stored-1 over
    /// <paramref name="members"/> devices with the default options.
    /// </summary>
    public long[] ExpectedMemberCounts(MultiGpuMode mode, int stored, int members) =>
        Enumerable.Range(0, members).Select(i => mode == MultiGpuMode.Replicated
            ? stored
            : IsFlat
                // clone_Index_to_shards: shard i gets [ntotal * i / n, ntotal * (i + 1) / n)
                ? (long)stored * (i + 1) / members - (long)stored * i / members
                // default shard type IdModulo: shard i gets the ids with id % n == i
                : Enumerable.Range(0, stored).LongCount(id => id % members == i)).ToArray();

    public override string ToString() => Name;
}

public static class MultiGpuCatalog
{
    private static readonly MultiGpuIndexCase[] Cases =
    [
        Case<IndexFlatL2, GpuIndexFlatL2>("FlatL2"),
        Case<IndexFlatIP, GpuIndexFlatIP>("FlatIP"),
        Case<IndexFlat, GpuIndexFlat>("FlatGenericL2"),
        Case<IndexFlat, GpuIndexFlat>("FlatGenericIP"),
        Case<IndexIVFFlat<IndexFlatL2>, GpuIndexIVFFlat<IndexFlatL2>>("IVFFlatL2"),
        Case<IndexIVFFlat<IndexFlatIP>, GpuIndexIVFFlat<IndexFlatIP>>("IVFFlatIP"),
        Case<IndexIVFScalarQuantizer<IndexFlatL2>, GpuIndexIVFScalarQuantizer<IndexFlatL2>>("IVFSQ8"),
        Case<IndexIVFScalarQuantizer<IndexFlatL2>, GpuIndexIVFScalarQuantizer<IndexFlatL2>>("IVFSQfp16"),
        Case<IndexIVFPQ, GpuIndexIVFPQ>("IVFPQ"),
    ];

    public static int[][] DeviceSets => FaissGpu.GetNumGpus() >= 2 ? [[0, 0], [0, 1], [1, 0]] : [[0, 0]];

    public static TheoryData<MultiGpuIndexCase, MultiGpuMode, int[]> AllOnEveryDeviceSet { get; } = Cross(_ => true);

    public static TheoryData<MultiGpuIndexCase, MultiGpuMode, int[]> ExactOnEveryDeviceSet { get; } = Cross(c => c.Single.Has(GpuCapabilities.Exact));

    public static TheoryData<MultiGpuIndexCase, MultiGpuMode> All { get; } = Modes(_ => true);

    public static TheoryData<MultiGpuIndexCase, MultiGpuMode> Approximate { get; } = Modes(c => !c.Single.Has(GpuCapabilities.Exact));

    public static TheoryData<MultiGpuIndexCase> Each { get; } = Each_(_ => true);

    public static MultiGpuIndexCase Get(string name) => Cases.Single(c => c.Name == name);

    private static MultiGpuIndexCase Case<TCpu, TGpu>(string name)
        where TCpu : FloatIndex, IMultiGpuClonableIndex<TCpu, TGpu>, IFromNativeIndexHandle<TCpu>
        where TGpu : FloatIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
        => new(GpuIndexCatalog.Get(name),
            (providers, devices, cpu, options) => GpuIndexProvider.TransferToGpuSharded(providers, devices, (TCpu)cpu, options),
            (providers, devices, cpu, options) => GpuIndexProvider.TransferToGpuReplicated(providers, devices, (TCpu)cpu, options),
            squad => squad switch
            {
                GpuIndexShards<TGpu, TCpu> shards => GpuIndexProvider.TransferToCpu(shards),
                GpuIndexReplicas<TGpu, TCpu> replicas => GpuIndexProvider.TransferToCpu(replicas),
                _ => throw new ArgumentException($"Not a multi-GPU index: {squad.GetType()}", nameof(squad)),
            },
            squad => squad switch
            {
                GpuIndexShards<TGpu, TCpu> shards => Enumerable.Range(0, shards.Count).Select(i => shards[i].TotalCount).ToArray(),
                GpuIndexReplicas<TGpu, TCpu> replicas => Enumerable.Range(0, replicas.Count).Select(i => replicas[i].TotalCount).ToArray(),
                _ => throw new ArgumentException($"Not a multi-GPU index: {squad.GetType()}", nameof(squad)),
            },
            squad => ((IGpuDevicesSquadIndex<TCpu>)squad).DeviceIds);

    private static TheoryData<MultiGpuIndexCase, MultiGpuMode, int[]> Cross(Func<MultiGpuIndexCase, bool> predicate)
    {
        var data = new TheoryData<MultiGpuIndexCase, MultiGpuMode, int[]>();
        foreach (var c in Cases.Where(predicate))
        foreach (var mode in Enum.GetValues<MultiGpuMode>())
        foreach (var devices in DeviceSets)
            data.Add(c, mode, devices);

        return data;
    }

    private static TheoryData<MultiGpuIndexCase, MultiGpuMode> Modes(Func<MultiGpuIndexCase, bool> predicate)
    {
        var data = new TheoryData<MultiGpuIndexCase, MultiGpuMode>();
        foreach (var c in Cases.Where(predicate))
        foreach (var mode in Enum.GetValues<MultiGpuMode>())
            data.Add(c, mode);

        return data;
    }

    private static TheoryData<MultiGpuIndexCase> Each_(Func<MultiGpuIndexCase, bool> predicate)
    {
        var data = new TheoryData<MultiGpuIndexCase>();
        foreach (var c in Cases.Where(predicate))
            data.Add(c);

        return data;
    }
}

/// <summary>One distinct provider per device.</summary>
/// <remarks>Faiss's resources are not thread safe, and every member runs on its own thread.</remarks>
public sealed class ProviderSet : IDisposable
{
    public ProviderSet(int count)
    {
        Items = new GpuResourcesProvider[count];
        for (var i = 0; i < count; i++)
            Items[i] = new GpuResourcesProvider();
    }

    public GpuResourcesProvider[] Items { get; }

    public void Dispose()
    {
        foreach (var provider in Items)
            provider.Dispose();
    }
}
