using Faiss.Gpu.Interfaces;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Interop.Errors;

namespace Faiss.Gpu.Resources;

/// <summary>
/// Provides scoped access to the native GPU resources obtained from a
/// <see cref="GpuResourcesProvider"/>.
/// </summary>
public ref struct GpuResources
{
    private readonly IGpuResourcesProvider _parent;

    private IntPtr Handle { get; }

    internal GpuResources(IGpuResourcesProvider parent, IntPtr handle)
    {
        _parent = parent;
        Handle = handle;
    }

    /// <summary>
    /// Pre-allocates GPU resources for the specified device.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    public void InitializeForDevice(int device)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_initializeForDevice(Handle, device)
        );
    }

    /// <summary>
    /// Retrieves the native BLAS handle for the specified device.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    /// <returns>An opaque pointer to the native BLAS handle.</returns>
    public IntPtr GetBlasHandle(int device)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getBlasHandle(Handle, device, out IntPtr handle)
        );
        return handle;
    }

    /// <summary>
    /// Retrieves the native BLAS handle for the currently active device.
    /// </summary>
    /// <returns>An opaque pointer to the native BLAS handle.</returns>
    public IntPtr GetBlasHandleCurrentDevice()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getBlasHandleCurrentDevice(Handle, out IntPtr handle)
        );
        return handle;
    }

    /// <summary>
    /// Retrieves the default computation stream for the specified device.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    /// <returns>An opaque pointer to the native stream.</returns>
    public IntPtr GetDefaultStream(int device)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getDefaultStream(Handle, device, out IntPtr stream)
        );
        return stream;
    }

    /// <summary>
    /// Retrieves the default computation stream for the currently active device.
    /// </summary>
    /// <returns>An opaque pointer to the native stream.</returns>
    public IntPtr GetDefaultStreamCurrentDevice()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getDefaultStreamCurrentDevice(Handle, out IntPtr stream)
        );
        return stream;
    }

    /// <summary>
    /// Retrieves the pinned host memory buffer available for asynchronous
    /// CPU-to-GPU and GPU-to-CPU transfers.
    /// </summary>
    /// <param name="ptr">The pointer to the pinned memory buffer.</param>
    /// <param name="size">The size of the buffer in bytes.</param>
    public void GetPinnedMemory(out IntPtr ptr, out nuint size)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getPinnedMemory(Handle, out ptr, out size)
        );
    }

    /// <summary>
    /// Retrieves the asynchronous copy stream for the specified device.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    /// <returns>An opaque pointer to the native stream.</returns>
    public IntPtr GetAsyncCopyStream(int device)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getAsyncCopyStream(Handle, device, out IntPtr stream)
        );
        return stream;
    }

    /// <summary>
    /// Retrieves the asynchronous copy stream for the currently active device.
    /// </summary>
    /// <returns>An opaque pointer to the native stream.</returns>
    public IntPtr GetAsyncCopyStreamCurrentDevice()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_getAsyncCopyStreamCurrentDevice(Handle, out IntPtr stream)
        );
        return stream;
    }

    /// <summary>
    /// Synchronizes the CPU with the default stream for the specified device.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    public void SyncDefaultStream(int device)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_syncDefaultStream(Handle, device)
        );
    }

    /// <summary>
    /// Synchronizes the CPU with the default stream for the currently active device.
    /// </summary>
    public void SyncDefaultStreamCurrentDevice()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResources_syncDefaultStreamCurrentDevice(Handle)
        );
    }
}