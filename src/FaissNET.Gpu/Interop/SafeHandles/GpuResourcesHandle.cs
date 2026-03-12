namespace Faiss.Gpu.Interop.SafeHandles;

using Microsoft.Win32.SafeHandles;

internal sealed class GpuResourcesHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public GpuResourcesHandle() : base(true)
    {
    }

    internal GpuResourcesHandle(IntPtr ptr) : base(true)
    {
        SetHandle(ptr);
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            GpuNativeMethods.faiss_StandardGpuResources_free(handle);
            handle = IntPtr.Zero;
        }
        return true;
    }
}