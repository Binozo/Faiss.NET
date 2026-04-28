using Faiss.Exceptions;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Interop.Errors;

namespace Faiss.Gpu;

/// <summary>
/// Provides global utilities for GPU-based operations in Faiss,
/// including device enumeration, synchronization, and profiler control.
/// </summary>
public static class FaissGpu
{
    /// <summary>
    /// Retrieves the number of CUDA/HIP-compatible GPU devices
    /// available to the Faiss library.
    /// </summary>
    /// <returns>The total number of accessible GPU devices.</returns>
    /// <exception cref="FaissException">
    /// Thrown when the native library encounters an error while querying devices.
    /// </exception>
    public static int GetNumGpus()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_get_num_gpus(out int count)
        );

        return count;
    }

    /// <summary>
    /// Synchronizes the CPU with all GPU devices by blocking until
    /// all preceding operations in the default streams have completed.
    /// </summary>
    /// <exception cref="FaissException">
    /// Thrown when the native synchronization call fails.
    /// </exception>
    public static void SyncAllDevices() => FaissErrorHandler.ThrowIfError(GpuNative.faiss_gpu_sync_all_devices());

    /// <summary>
    /// Signals the external GPU profiler (e.g., NVIDIA Nsight Systems)
    /// to begin recording device activity.
    /// </summary>
    /// <remarks>
    /// This method is a no-op if no external profiler is attached.
    /// It is typically used to bracket performance-critical code regions
    /// when profiling with <c>nsys</c> or legacy <c>nvprof</c>.
    /// </remarks>
    /// <exception cref="FaissException">
    /// Thrown when the underlying native profiler call fails.
    /// </exception>
    public static void StartProfiler() => FaissErrorHandler.ThrowIfError(GpuNative.faiss_gpu_profiler_start());

    /// <summary>
    /// Signals the external GPU profiler to stop recording device activity.
    /// </summary>
    /// <remarks>
    /// Must be paired with a preceding call to <see cref="StartProfiler"/>.
    /// </remarks>
    /// <exception cref="FaissException">
    /// Thrown when the underlying native profiler call fails.
    /// </exception>
    public static void StopProfiler() => FaissErrorHandler.ThrowIfError(GpuNative.faiss_gpu_profiler_stop());
}