using Faiss.Cpu.Indexes;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Interop;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Gpu.Resources;

namespace Faiss.Gpu;

using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;

/// <summary>
/// Provides factory methods for transferring Faiss indexes between CPU and GPU memory.
/// </summary>
public static class GpuIndexProvider
{
    /// <summary>
    /// Transfers a CPU index to the specified GPU device.
    /// </summary>
    /// <typeparam name="TCpu">The type of the CPU index.</typeparam>
    /// <typeparam name="TGpu">The type of the GPU index.</typeparam>
    /// <param name="context">The GPU resource provider that manages memory and streams.</param>
    /// <param name="cpuIndex">The CPU index to transfer.</param>
    /// <param name="deviceId">The target GPU device ID. Defaults to <c>0</c>.</param>
    /// <returns>A <see cref="GpuIndex{T}"/> that resides on the specified GPU device.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="context"/> or <paramref name="cpuIndex"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="deviceId"/> is negative or not below <see cref="FaissGpu.GetNumGpus"/>.
    /// </exception>
    /// <exception cref="FaissGpuCloningUnsupported">
    /// Thrown when faiss cannot place this index, as configured, on a GPU.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static TGpu TransferToGpu<TCpu, TGpu>(GpuResourcesProvider context, IGpuClonableIndex<TCpu, TGpu> cpuIndex, int deviceId = 0)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);
        GpuDevice.ThrowIfInvalid(deviceId);

        if (!cpuIndex.IsGpuClonable())
            throw new FaissGpuCloningUnsupported();

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu(
            context.Handle,
            deviceId,
            cpuIndex.Handle,
            out IntPtr gpuHandle));

        var gpuIndex = TGpu.FromPointer(gpuHandle);
        cpuIndex.OnClonedToGpu(gpuIndex);
        return gpuIndex;
    }

    /// <summary>
    /// Transfers a CPU index to the specified GPU device using advanced cloning options.
    /// </summary>
    /// <typeparam name="TCpu">The type of the CPU index.</typeparam>
    /// <typeparam name="TGpu">The type of the GPU index.</typeparam>
    /// <param name="context">The GPU resource provider that manages memory and streams.</param>
    /// <param name="cpuIndex">The CPU index to transfer.</param>
    /// <param name="options">Options that control how the index is cloned onto the GPU.</param>
    /// <param name="deviceId">The target GPU device ID. Defaults to <c>0</c>.</param>
    /// <returns>A <see cref="TGpu"/> that resides on the specified GPU device.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="context"/>, <paramref name="cpuIndex"/>, or <paramref name="options"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="deviceId"/> is negative or not below <see cref="FaissGpu.GetNumGpus"/>.
    /// </exception>
    /// <exception cref="FaissGpuCloningUnsupported">
    /// Thrown when faiss cannot place this index, as configured, on a GPU.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="options"/> asks for <see cref="IndicesOptions.Bit32"/> storage and the IVF
    /// <paramref name="cpuIndex"/> holds an id outside the 32-bit range.
    /// </exception>
    public static TGpu TransferToGpu<TCpu, TGpu>(GpuResourcesProvider context, IGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuClonerOptions options, int deviceId = 0)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);
        ArgumentNullException.ThrowIfNull(options);
        GpuDevice.ThrowIfInvalid(deviceId);

        if (!cpuIndex.IsGpuClonable())
            throw new FaissGpuCloningUnsupported();

        Bit32Ids.ThrowIfSourceHoldsWideIds(cpuIndex, options);

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu_with_options(
            context.Handle,
            deviceId,
            cpuIndex.Handle,
            options.NativeHandle,
            out IntPtr gpuHandle));

        var gpuIndex = TGpu.FromPointer(gpuHandle);
        cpuIndex.OnClonedToGpu(gpuIndex);
        Bit32Ids.Mark(gpuIndex, options);
        return gpuIndex;
    }

    /// <inheritdoc cref="TransferToGpuMultiple"/>
    public static GpuIndexReplicas<TGpu, TCpu> TransferToGpuReplicated<TCpu, TGpu>(GpuResourcesProvider[] contexts, int[] deviceIds, IMultiGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuMultipleClonerOptions? options = null)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
        where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
    {
        GpuMultipleClonerOptions multiOptions = options ?? new GpuMultipleClonerOptions();

        try
        {
            multiOptions.Shard = false;
            var gpuIndex = GpuIndexReplicas<TGpu, TCpu>.FromPointer(TransferToGpuMultiple(contexts, deviceIds, cpuIndex, multiOptions));
            cpuIndex.OnClonedToGpu(gpuIndex);
            return gpuIndex;
        }
        finally
        {
            if(options == null)
                multiOptions.Dispose();
        }
    }

    /// <inheritdoc cref="TransferToGpuMultiple"/>
    /// <remarks>
    /// The index is split across the devices. Every device holds a part of the vectors and a search merges their results.
    /// <para>
    /// <b>Warning:</b> faiss builds the shards in parallel inside an OpenMP region, where a native error cannot reach the caller
    /// and terminates the process instead. Every condition that can be checked in advance is rejected before faiss is called
    /// (<see cref="FaissGpuCloningUnsupported"/>, <see cref="ArgumentException"/>), but runtime failures cannot be:
    /// running out of device memory (or pinned host memory) while a shard is built kills the process. Leave headroom on every
    /// target device. When memory is tight, prefer <see cref="TransferToGpuReplicated{TCpu, TGpu}"/> or
    /// <see cref="TransferToGpu{TCpu, TGpu}(GpuResourcesProvider, IGpuClonableIndex{TCpu, TGpu}, int)"/>, which report the same failure as a
    /// <see cref="FaissNativeException"/>.
    /// </para>
    /// </remarks>
    public static GpuIndexShards<TGpu, TCpu> TransferToGpuSharded<TCpu, TGpu>(GpuResourcesProvider[] contexts, int[] deviceIds, IMultiGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuMultipleClonerOptions? options = null)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
        where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
    {
        GpuMultipleClonerOptions multiOptions = options ?? new GpuMultipleClonerOptions();
        
        try
        {
            multiOptions.Shard = true;
            var gpuIndex = GpuIndexShards<TGpu, TCpu>.FromPointer(TransferToGpuMultiple(contexts, deviceIds, cpuIndex, multiOptions));
            cpuIndex.OnClonedToGpu(gpuIndex);
            Bit32Ids.Mark(gpuIndex, multiOptions);
            return gpuIndex;
        }
        finally
        {
            if(options == null)
                multiOptions.Dispose();
        }
    }

    /// <summary>
    /// Transfers a CPU index across multiple GPU devices using advanced cloning options.
    /// </summary>
    /// <remarks>
    /// Depending on the <paramref name="options"/>, the index may be sharded
    /// (split across devices) or replicated (copied to each device).
    /// </remarks>
    /// <typeparam name="TCpu">The type of the CPU index.</typeparam>
    /// <param name="contexts">
    /// An array of GPU resource providers, one per target device.
    /// </param>
    /// <param name="deviceIds">
    /// An array of GPU device IDs. Must be the same length as <paramref name="contexts"/>.
    /// </param>
    /// <param name="cpuIndex">The CPU index to transfer.</param>
    /// <param name="options">
    /// Options that control how the index is cloned and distributed across GPUs.
    /// </param>
    /// <returns>A pointer to the index spanning the specified devices.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="contexts"/>, <paramref name="deviceIds"/>,
    /// or <paramref name="cpuIndex"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="contexts"/> and <paramref name="deviceIds"/>
    /// have different lengths, or when <paramref name="contexts"/> is empty.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    private static IntPtr TransferToGpuMultiple<TCpu, TGpu>(GpuResourcesProvider[] contexts, int[] deviceIds, IMultiGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuMultipleClonerOptions options)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(contexts);
        ArgumentNullException.ThrowIfNull(deviceIds);
        ArgumentNullException.ThrowIfNull(cpuIndex);

        if (contexts.Length == 0 || contexts.Length != deviceIds.Length)
            throw new ArgumentException("The contexts and deviceIds arrays must be non-empty and of equal length.", nameof(contexts));
        if(deviceIds.Length < 2)
            throw new ArgumentException("At least two target devices must be specified.", nameof(deviceIds));

        foreach (var context in contexts)
            ArgumentNullException.ThrowIfNull(context, nameof(contexts));

        if (contexts.Distinct(ReferenceEqualityComparer.Instance).Count() != contexts.Length)
            throw new ArgumentException("Every target device needs its own GpuResourcesProvider; the same instance appears more than once.", nameof(contexts));

        if (!cpuIndex.IsMultiGpuClonable())
            throw new FaissGpuCloningUnsupported();

        Bit32Ids.ThrowIfSourceHoldsWideIds(cpuIndex, options);

        if (options.Shard)
            ShardedClone.ThrowIfUnsupported(cpuIndex, options, deviceIds.Length);

        int count = contexts.Length;
        bool[] addedRefs = new bool[count];
        IntPtr[] providerPointers = new IntPtr[count];

        try
        {
            for (int i = 0; i < count; i++)
            {
                GpuDevice.ThrowIfInvalid(deviceIds[i], nameof(deviceIds));

                contexts[i].Handle.DangerousAddRef(ref addedRefs[i]);
                providerPointers[i] = contexts[i].Handle.DangerousGetHandle();
            }

            unsafe
            {
                fixed (IntPtr* pProviders = providerPointers)
                fixed (int* pDevices = deviceIds)
                {
                    FaissErrorHandler.ThrowIfError(
                        GpuNative.faiss_index_cpu_to_gpu_multiple_with_options(
                            pProviders,
                            (nuint)count,
                            pDevices,
                            (nuint)count,
                            cpuIndex.Handle,
                            options.NativeHandle,
                            out IntPtr gpuHandle)
                    );

                    return gpuHandle;
                }
            }
        }
        finally
        {
            for (int i = 0; i < count; i++)
            {
                if (addedRefs[i])
                    contexts[i].Handle.DangerousRelease();
            }
        }
    }

    /// <summary>
    /// Transfers a GPU index back to CPU memory.
    /// </summary>
    /// <typeparam name="T">The type of the resulting CPU index.</typeparam>
    /// <param name="gpuIndex">The GPU index to transfer back.</param>
    /// <returns>A CPU index reconstructed from the GPU index.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="gpuIndex"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static T TransferToCpu<T>(IGpuIndex<T> gpuIndex) where T : INativeIndex, IFromNativeIndexHandle<T>
    {
        ArgumentNullException.ThrowIfNull(gpuIndex);

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_gpu_to_cpu(gpuIndex.Handle, out IntPtr ptr));

        return T.FromPointer(ptr);
    }
}