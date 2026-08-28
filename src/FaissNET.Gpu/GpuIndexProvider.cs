using Faiss.Cpu.Indexes;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
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
    /// <exception cref="FaissException">
    /// <exception cref="FaissGpuCloningUnsupported">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static TGpu TransferToGpu<TCpu, TGpu>(GpuResourcesProvider context, IGpuClonableIndex<TCpu, TGpu> cpuIndex, int deviceId = 0)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);

        if (!cpuIndex.IsGpuClonable())
        {
            throw new FaissGpuCloningUnsupported();
        }

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu(
            context.Handle,
            deviceId,
            cpuIndex.Handle,
            out IntPtr gpuHandle));

        // Currently there is no way of calling the c api to know on which device the index is on
        // It is only available in C++ and not wrapped in C, the user will have to keep track himself
        return TGpu.FromPointer(gpuHandle);
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
    /// <returns>A <see cref="GpuIndex{T}"/> that resides on the specified GPU device.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="context"/>, <paramref name="cpuIndex"/>, or <paramref name="options"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static TGpu TransferToGpu<TCpu, TGpu>(GpuResourcesProvider context, IGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuClonerOptions options, int deviceId = 0)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);
        ArgumentNullException.ThrowIfNull(options);

        if (!cpuIndex.IsGpuClonable())
        {
            throw new FaissGpuCloningUnsupported();
        }

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu_with_options(
            context.Handle,
            deviceId,
            cpuIndex.Handle,
            options.NativeHandle,
            out IntPtr gpuHandle));

        return TGpu.FromPointer(gpuHandle);
    }

    /// <summary>
    /// Transfers a CPU index across multiple GPU devices using advanced cloning options.
    /// </summary>
    /// <remarks>
    /// Depending on the <paramref name="options"/>, the index may be sharded
    /// (split across devices) or replicated (copied to each device).
    /// </remarks>
    /// <typeparam name="T">The type of the CPU index.</typeparam>
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
    /// <returns>A <see cref="GpuShardedIndex{T}"/> spanning the specified devices.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="contexts"/>, <paramref name="deviceIds"/>,
    /// <paramref name="cpuIndex"/>, or <paramref name="options"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="contexts"/> and <paramref name="deviceIds"/>
    /// have different lengths, or when <paramref name="contexts"/> is empty.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static TGpu TransferToGpuMultiple<TCpu, TGpu>(GpuResourcesProvider[] contexts, int[] deviceIds, IMultiGpuClonableIndex<TCpu, TGpu> cpuIndex, GpuMultipleClonerOptions options)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(contexts);
        ArgumentNullException.ThrowIfNull(deviceIds);
        ArgumentNullException.ThrowIfNull(cpuIndex);
        ArgumentNullException.ThrowIfNull(options);

        if (contexts.Length == 0 || contexts.Length != deviceIds.Length)
        {
            throw new ArgumentException(
                "The contexts and deviceIds arrays must be non-empty and of equal length.",
                nameof(contexts));
        }

        if (!cpuIndex.IsMultiGpuClonable())
        {
            throw new FaissGpuCloningUnsupported();
        }

        int count = contexts.Length;
        IntPtr[] providerPointers = new IntPtr[count];

        for (int i = 0; i < count; i++)
        {
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

                return TGpu.FromPointer(gpuHandle);
            }
        }
    }

    /// <summary>
    /// Transfers a CPU index across multiple GPU devices using default cloning options.
    /// </summary>
    /// <remarks>
    /// Depending on the default behavior of the native library, the index may be
    /// sharded or replicated across the specified devices.
    /// </remarks>
    /// <typeparam name="T">The type of the CPU index.</typeparam>
    /// <param name="contexts">
    /// An array of GPU resource providers, one per target device.
    /// </param>
    /// <param name="deviceIds">
    /// An array of GPU device IDs. Must be the same length as <paramref name="contexts"/>.
    /// </param>
    /// <param name="cpuIndex">The CPU index to transfer.</param>
    /// <returns>A <see cref="GpuShardedIndex{T}"/> spanning the specified devices.</returns>
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
    public static TGpu TransferToGpuMultiple<TCpu, TGpu>(GpuResourcesProvider[] contexts, int[] deviceIds, IMultiGpuClonableIndex<TCpu, TGpu> cpuIndex)
        where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
    {
        ArgumentNullException.ThrowIfNull(contexts);
        ArgumentNullException.ThrowIfNull(deviceIds);
        ArgumentNullException.ThrowIfNull(cpuIndex);

        if (contexts.Length == 0 || contexts.Length != deviceIds.Length)
        {
            throw new ArgumentException(
                "The contexts and deviceIds arrays must be non-empty and of equal length.",
                nameof(contexts));
        }

        int count = contexts.Length;
        IntPtr[] providerPointers = new IntPtr[count];

        for (int i = 0; i < count; i++)
        {
            providerPointers[i] = contexts[i].Handle.DangerousGetHandle();
        }

        unsafe
        {
            fixed (IntPtr* pProviders = providerPointers)
            fixed (int* pDevices = deviceIds)
            {
                FaissErrorHandler.ThrowIfError(
                    GpuNative.faiss_index_cpu_to_gpu_multiple(
                        pProviders,
                        pDevices,
                        (nuint)count,
                        cpuIndex.Handle,
                        out IntPtr gpuHandle)
                );

                return TGpu.FromPointer(gpuHandle);
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

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_gpu_to_cpu(
            gpuIndex.Handle,
            out IntPtr ptr));

        return T.FromPointer(ptr);
    }
}