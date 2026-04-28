using Faiss.Gpu.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Gpu.Interop.SafeHandles;

internal sealed class GpuResourcesProviderHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public GpuResourcesProviderHandle(IntPtr handle) : base(true)
    {
        SetHandle(handle);
    }

    protected override bool ReleaseHandle()
    {
        GpuNative.faiss_StandardGpuResources_free(handle);
        return true;
    }
}