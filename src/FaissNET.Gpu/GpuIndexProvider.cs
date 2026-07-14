using Faiss.Cpu.Indexes;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Indexes;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Gpu.Resources;

namespace Faiss.Gpu;

using Faiss.Cpu.Interfaces;
using Interfaces;
using Faiss.Interop.Errors;

/// <summary>
/// Provides factory methods for transferring Faiss indexes between CPU and GPU memory.
/// </summary>
public static class GpuIndexProvider
{
    /// <summary>
    /// Transfers a CPU index to the specified GPU device.
    /// </summary>
    /// <typeparam name="T">The type of the CPU index.</typeparam>
    /// <param name="context">The GPU resource provider that manages memory and streams.</param>
    /// <param name="cpuIndex">The CPU index to transfer.</param>
    /// <param name="deviceId">The target GPU device ID. Defaults to <c>0</c>.</param>
    /// <returns>A <see cref="GpuIndex{T}"/> that resides on the specified GPU device.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="context"/> or <paramref name="cpuIndex"/> is <c>null</c>.
    /// </exception>
    /// <exception cref="FaissException">
    /// Thrown when the native transfer operation fails.
    /// </exception>
    public static INativeGpuIndex<T> TransferToGpu<T>(GpuResourcesProvider context, T cpuIndex, int deviceId = 0) where T : CpuIndex<T>, IFromNativeHandle<T>, IGpuClonableIndex
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu(
            context.Handle,
            deviceId,
            cpuIndex.SafeHandle,
            out IntPtr gpuHandle));

        return new GpuIndex<T>(gpuHandle, deviceId);
    }

    /// <summary>
    /// Transfers a CPU index to the specified GPU device using advanced cloning options.
    /// </summary>
    /// <typeparam name="T">The type of the CPU index.</typeparam>
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
    public static INativeGpuIndex<T> TransferToGpu<T>(GpuResourcesProvider context, T cpuIndex, GpuClonerOptions options, int deviceId = 0) where T : CpuIndex<T>, IFromNativeHandle<T>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(cpuIndex);
        ArgumentNullException.ThrowIfNull(options);

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_cpu_to_gpu_with_options(
            context.Handle,
            deviceId,
            cpuIndex.SafeHandle,
            options.NativeHandle,
            out IntPtr gpuHandle));

        return new GpuIndex<T>(gpuHandle, deviceId);
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
    public static GpuShardedIndex<T> TransferToGpuMultiple<T>(GpuResourcesProvider[] contexts, int[] deviceIds, T cpuIndex, GpuMultipleClonerOptions options) where T : Index<T>, IFromNativeHandle<T>
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
                        cpuIndex.SafeHandle,
                        options.NativeHandle,
                        out IntPtr gpuHandle)
                );

                return new GpuShardedIndex<T>(gpuHandle, deviceIds);
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
    public static GpuShardedIndex<T> TransferToGpuMultiple<T>(GpuResourcesProvider[] contexts, int[] deviceIds, T cpuIndex) where T : Index<T>, IFromNativeHandle<T>
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
                        cpuIndex.SafeHandle,
                        out IntPtr gpuHandle)
                );

                return new GpuShardedIndex<T>(gpuHandle, deviceIds);
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
    public static T TransferToCpu<T>(INativeGpuIndex<T> gpuIndex) where T : CpuIndex<T>, IFromNativeHandle<T>
    {
        ArgumentNullException.ThrowIfNull(gpuIndex);

        FaissErrorHandler.ThrowIfError(GpuNative.faiss_index_gpu_to_cpu(
            gpuIndex.Handle,
            out IntPtr cpuHandle));

        return T.FromHandle(cpuHandle);
    }
}