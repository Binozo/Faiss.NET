using System.Numerics;
using Faiss.Models;

namespace Faiss.Cpu.Interfaces;

public interface IGpuIndex : INativeIndex
{
}

public interface IGpuIndex<T> : IGpuIndex where T : INativeIndex, IFromNativeIndexHandle<T>
{
}

public interface IGpuDeviceIndex : IGpuIndex
{
}

public interface IGpuDeviceIndex<T> : IGpuDeviceIndex, IGpuIndex<T> where T : INativeIndex, IFromNativeIndexHandle<T>
{
}

public interface IGpuDevicesSquadIndex<T> : IGpuIndex<T> where T : INativeIndex, IFromNativeIndexHandle<T>
{
    public int[] DeviceIds { get; }
}

internal interface IGpuIVFIndex : IGpuIndex
{
    /// <summary>
    /// Set when the index was transferred with 32-bit id storage. Faiss then truncates every id added on the device to
    /// 32 bits without checking, so the wrapper has to reject wider ids itself.
    /// </summary>
    bool Uses32BitIds { get; set; }
}

internal static class GpuIVFIDs
{
    public static void ThrowIfNot32Bit(IGpuIVFIndex index, ReadOnlySpan<long> xids)
    {
        if (!index.Uses32BitIds)
            return;

        foreach (var id in xids)
        {
            if (id is < int.MinValue or > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(xids), id, "The index stores ids in 32 bits (IndicesOptions.Bit32); this id does not fit.");
        }
    }
}

internal interface IGpuScalarQuantizerIndex : IGpuIndex
{
    QuantizerType? QuantizerType { get; set; }
}

internal static class GpuScalarQuantizerLimits
{
    /// <summary>
    /// Device QT_8bit and QT_4bit IVF scalar quantizers need fewer dimensions than this (GpuIndexIVFScalarQuantizer::verifySQSettings_).
    /// </summary>
    /// <remarks>
    /// The codec keeps vmin and vdiff, two floats, per dimension in shared memory, and faiss requires
    /// d &lt; getMaxSharedMemPerBlock(device) / (2 * sizeof(float)). getMaxSharedMemPerBlock returns
    /// cudaDeviceProp::sharedMemPerBlock, which is 48 KiB on every CUDA device: 49152 / 8 = 6144.
    /// faiss checks this while it builds the device index, which a sharded transfer does inside an OpenMP region where
    /// the throw terminates the process, so it is checked up front.
    /// </remarks>
    public const int MaxPerDimensionCodecDimensions = 49152 / (2 * sizeof(float));

    /// <summary>Whether faiss can build a device index for this codec at all.</summary>
    public static bool FitsDevice(QuantizerType quantizerType, long dimensions) =>
        quantizerType is not (QuantizerType.QT_8bit or QuantizerType.QT_4bit) || dimensions < MaxPerDimensionCodecDimensions;

    /// <summary>Whether the device scan of <paramref name="quantizerType"/> keeps per-dimension ranges in shared memory.</summary>
    public static bool LimitsK(QuantizerType? quantizerType) => quantizerType is QuantizerType.QT_8bit or QuantizerType.QT_4bit or QuantizerType.QT_6bit;

    /// <summary>
    /// The quantizer a squad keeps after <paramref name="member"/> joined.
    /// A limiting one wins, since every limiting type shares the same launch bound.
    /// </summary>
    public static QuantizerType? Combine(QuantizerType? current, INativeIndex member) => LimitsK(current) ? current : (member as IGpuScalarQuantizerIndex)?.QuantizerType ?? current;

    public static void ThrowIfKUnsupported(QuantizerType? quantizerType, long dimensions, long k)
    {
        if (!LimitsK(quantizerType))
            return;

        // ivfInterleavedScan: 128 threads (4 warps) each keep a queue of k rounded up to a power of two (at least 32)
        // of float distances and 64-bit ids in static shared memory; the codec adds vmin and vdiff per dimension.
        // Queues above 512 entries already fill the 48 KiB launch limit on their own.
        const long sharedMemoryLimit = 48 * 1024;
        var queue = Math.Max(32L, (long)BitOperations.RoundUpToPowerOf2((ulong)k));
        var required = 4 * queue * (sizeof(float) + sizeof(long)) + 2L * sizeof(float) * dimensions;

        if (queue > 512 || required > sharedMemoryLimit)
            throw new ArgumentOutOfRangeException(nameof(k), k, $"A GPU {quantizerType} scalar quantizer index with {dimensions} dimensions cannot search for {k} neighbours: the device scan needs {required} bytes of shared memory, the limit is {sharedMemoryLimit}.");
    }
}
