namespace Faiss.Gpu;

using Faiss.Interop.ErrorHandling;
using Interop;
using Interop.SafeHandles;

public sealed class FaissGpuContext : IDisposable
{
    internal IntPtr ResourcePointer { get; }
    
    private readonly GpuResourcesHandle _handle;

    public FaissGpuContext()
    {
        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_StandardGpuResources_new(out IntPtr ptr));
        ResourcePointer = ptr;

        _handle = new GpuResourcesHandle(ptr);
    }

    /// <summary>
    /// Sets the absolute limit for temporary scratch space in VRAM. 
    /// </summary>
    public void SetTempMemory(nuint sizeInBytes)
    {
        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_StandardGpuResources_setTempMemory(ResourcePointer, sizeInBytes));
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}