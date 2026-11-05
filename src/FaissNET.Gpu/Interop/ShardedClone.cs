using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Interop.NativeMethods;

namespace Faiss.Gpu.Interop;

internal static class ShardedClone
{
    /// <summary>Above this many sub-quantizers the float32 lookup tables of a device IVFPQ exceed 48 KiB of shared memory.</summary>
    private const int MaxFloat32LookupSubQuantizers = 48;

    /// <summary>
    /// <c>GPU_MAX_SELECTION_K</c> of the CUDA builds (<c>DeviceDefs.cuh</c>).
    /// </summary>
    /// <remarks>
    /// Some ROCm targets may use 1024 which is not covered here.
    /// </remarks>
    private const int MaxDeviceNprobe = 2048;

    public static void ThrowIfUnsupported(INativeIndex cpuIndex, GpuMultipleClonerOptions options, int shardCount)
    {
        // INDICES_IVF reports (list << 32 | offset) instead of ids. IVF shards hold disjoint parts of the same lists and their
        // labels are merged untranslated (successive_ids = false), so equal labels from different shards cannot be told apart.
        if (options.StorageMode == IndicesOptions.Ivf && cpuIndex is IIVFIndex)
            throw new ArgumentException($"{nameof(IndicesOptions)}.{nameof(IndicesOptions.Ivf)} cannot be sharded: every shard reports (list, offset) labels that collide with the other shards'. Use {nameof(IndicesOptions.Cpu)} or a 32/64-bit mode, or a replicated transfer.", nameof(options));

        if (cpuIndex is IIVFIndex { Nprobe: <= 0 or > MaxDeviceNprobe } source)
            throw new FaissGpuCloningUnsupported($"A GPU IVF index supports an nprobe between 1 and {MaxDeviceNprobe}; the index has {source.Nprobe}. Lower {nameof(IIVFIndex.Nprobe)} before sharding.");

        if (cpuIndex is IndexIVFPQ ivfpq && !options.UseFloat16 && ivfpq.SubQuantizerCount > MaxFloat32LookupSubQuantizers)
            throw new FaissGpuCloningUnsupported($"An IndexIVFPQ with more than {MaxFloat32LookupSubQuantizers} sub-quantizers needs {nameof(GpuClonerOptions.UseFloat16)} lookup tables on the GPU.");

        if (options.ShardType == GpuShardType.IdRange && cpuIndex is IIVFIndex ivf)
            ThrowIfIdsNotSequential(cpuIndex, ivf, shardCount);
    }

    /// <summary>
    /// <see cref="GpuShardType.IdRange"/> gives shard <c>s</c> the ids in <c>[s * ntotal / n, (s + 1) * ntotal / n)</c> and asserts
    /// that it received exactly that many (<c>copy_ivf_shard</c>), which only holds for the ids <c>0..ntotal-1</c>.
    /// </summary>
    private static unsafe void ThrowIfIdsNotSequential(INativeIndex cpuIndex, IIVFIndex ivf, int shardCount)
    {
        var handle = cpuIndex.Handle;
        var total = Native.faiss_Index_ntotal(handle);
        var perShard = new long[shardCount];
        long[] ids = [];

        for (var list = 0; list < ivf.Nlist; list++)
        {
            var size = (int)Native.faiss_IndexIVF_get_list_size(handle, (nuint)list);
            if (size == 0)
                continue;

            if (ids.Length < size)
                ids = new long[size];

            fixed (long* pIds = ids)
                Native.faiss_IndexIVF_invlists_get_ids(handle, (nuint)list, (IntPtr)pIds);

            foreach (var id in ids.AsSpan(0, size))
            {
                if (id < 0 || id >= total)
                    throw NotSequential(id, nameof(cpuIndex));

                // floor(id * n / total) never overshoots. Its range starts at or below id.
                // It can undershoot when total is not a multiple of n, so step up to the range that holds id.
                var shard = (int)(id * shardCount / total);
                while (id >= Start(shard + 1, total, shardCount))
                    shard++;

                perShard[shard]++;
            }
        }

        for (var shard = 0; shard < shardCount; shard++)
        {
            if (perShard[shard] != Start(shard + 1, total, shardCount) - Start(shard, total, shardCount))
                throw NotSequential(null, nameof(cpuIndex));
        }
    }

    private static long Start(int shard, long total, int shardCount) => shard * total / shardCount;

    private static ArgumentException NotSequential(long? id, string paramName) => new(
        $"{nameof(GpuShardType)}.{nameof(GpuShardType.IdRange)} needs the ids 0..ntotal-1, each exactly once"
        + (id is { } value ? $"; the index holds id {value}" : "; the index holds duplicate ids")
        + $". Use {nameof(GpuShardType)}.{nameof(GpuShardType.IdModulo)} or {nameof(GpuShardType.InvertedList)} instead.",
        paramName);
}
