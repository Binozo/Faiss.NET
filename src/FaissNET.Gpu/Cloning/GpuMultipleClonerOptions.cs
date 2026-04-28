using System.Runtime.InteropServices;
using Faiss.Exceptions;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Gpu.Interop.SafeHandles;
using Faiss.Interop.Errors;

namespace Faiss.Gpu.Cloning;

/// <summary>
/// Provides advanced cloning options for distributing a Faiss index
/// across multiple GPU devices.
/// </summary>
/// <remarks>
/// Inherits all single-GPU options from <see cref="GpuClonerOptions"/>
/// and adds sharding or replication control for multi-GPU setups.
/// </remarks>
public sealed class GpuMultipleClonerOptions : GpuClonerOptions
{
    internal new GpuMultipleClonerOptionsHandle NativeHandle => (GpuMultipleClonerOptionsHandle)GetUnderlyingHandle();

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuMultipleClonerOptions"/> class.
    /// </summary>
    /// <exception cref="FaissException">
    /// Thrown when the native options allocation fails.
    /// </exception>
    public GpuMultipleClonerOptions() : base(CreateHandle())
    {
    }

    private static SafeHandle CreateHandle()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuMultipleClonerOptions_new(out IntPtr ptr)
        );

        return new GpuMultipleClonerOptionsHandle(ptr);
    }

    /// <summary>
    /// Whether to shard the index across GPUs (true) or replicate it
    /// across all GPUs (false).
    /// </summary>
    public bool Shard
    {
        get => GpuNative.faiss_GpuMultipleClonerOptions_shard(NativeHandle) != 0;
        set => GpuNative.faiss_GpuMultipleClonerOptions_set_shard(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// The sharding strategy used when <see cref="Shard"/> is true.
    /// Corresponds to the IVF copy-subset type.
    /// </summary>
    public GpuShardType ShardType
    {
        get => (GpuShardType)GpuNative.faiss_GpuMultipleClonerOptions_shard_type(NativeHandle);
        set => GpuNative.faiss_GpuMultipleClonerOptions_set_shard_type(NativeHandle, (int)value);
    }
}