using Faiss.Exceptions;
using Faiss.Gpu.Interfaces;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Gpu.Interop.SafeHandles;
using Faiss.Interop.Errors;

namespace Faiss.Gpu.Resources;

/// <summary>
/// Manages GPU resources for Faiss, including temporary memory pools,
/// pinned host memory, and CUDA/HIP stream configuration.
/// </summary>
/// <remarks>
/// This type wraps the native <c>StandardGpuResources</c> object,
/// which is the default implementation of <see cref="IGpuResourcesProvider"/>.
/// </remarks>
public sealed class GpuResourcesProvider : IGpuResourcesProvider, IDisposable
{
    internal readonly GpuResourcesProviderHandle Handle;

    IntPtr IGpuResourcesProvider.NativeHandle => Handle.DangerousGetHandle();

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuResourcesProvider"/> class,
    /// allocating the underlying native GPU resource provider.
    /// </summary>
    /// <exception cref="FaissException">
    /// Thrown when the native resource allocation fails.
    /// </exception>
    public GpuResourcesProvider()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_new(out IntPtr ptr)
        );

        Handle = new GpuResourcesProviderHandle(ptr);
    }

    /// <summary>
    /// Disables temporary memory pooling. All temporary GPU allocations
    /// will use direct device allocation and deallocation.
    /// </summary>
    /// <remarks>
    /// This reduces memory overhead but may decrease performance due to
    /// the cost of repeated <c>hipMalloc</c>/<c>hipFree</c> (or CUDA equivalent) calls.
    /// </remarks>
    /// <exception cref="FaissException">
    /// Thrown when the native call fails.
    /// </exception>
    public void NoTempMemory()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_noTempMemory(Handle)
        );
    }

    /// <summary>
    /// Sets the size of the temporary GPU memory pool.
    /// </summary>
    /// <param name="sizeInBytes">
    /// The pool size in bytes. Must be non-negative.
    /// </param>
    /// <exception cref="FaissException">
    /// Thrown when the native call fails.
    /// </exception>
    public void SetTempMemory(long sizeInBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sizeInBytes);

        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_setTempMemory(Handle, (nuint)sizeInBytes)
        );
    }

    /// <summary>
    /// Allocates page-locked (pinned) host memory for asynchronous
    /// CPU-to-GPU and GPU-to-CPU transfers.
    /// </summary>
    /// <param name="sizeInBytes">
    /// The amount of pinned memory to allocate in bytes. Must be non-negative.
    /// </param>
    /// <exception cref="FaissException">
    /// Thrown when the native call fails.
    /// </exception>
    public void SetPinnedMemory(long sizeInBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sizeInBytes);
        
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_setPinnedMemory(Handle, (nuint)sizeInBytes)
        );
    }

    /// <summary>
    /// Associates a specific CUDA/HIP stream with a GPU device for
    /// subsequent Faiss operations.
    /// </summary>
    /// <param name="device">The target GPU device ID.</param>
    /// <param name="stream">
    /// A pointer to a CUDA/HIP stream. Use <see cref="IntPtr.Zero"/>
    /// to specify the default (null) stream.
    /// </param>
    /// <exception cref="FaissException">
    /// Thrown when the native call fails.
    /// </exception>
    public void SetDefaultStream(int device, IntPtr stream)
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_setDefaultStream(Handle, device, stream)
        );
    }

    /// <summary>
    /// Resets the default stream for all GPU devices to the null stream.
    /// </summary>
    /// <exception cref="FaissException">
    /// Thrown when the native call fails.
    /// </exception>
    public void SetDefaultNullStreamAllDevices()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_StandardGpuResources_setDefaultNullStreamAllDevices(Handle)
        );
    }

    /// <summary>
    /// Obtains a scoped reference to the underlying native GPU resources.
    /// </summary>
    /// <returns>
    /// A <see cref="GpuResources"/> that is valid only for the current
    /// stack frame and guaranteed not to outlive this provider.
    /// </returns>
    public GpuResources GetResources()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuResourcesProvider_getResources(Handle, out IntPtr resPtr)
        );

        return new GpuResources(this, resPtr);
    }

    /// <summary>
    /// Releases the native GPU resources held by this instance.
    /// </summary>
    public void Dispose()
    {
        Handle.Dispose();
        GC.SuppressFinalize(this);
    }
}