using Faiss.Gpu.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Gpu.Interop.SafeHandles;

internal class GpuClonerOptionsHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal GpuClonerOptionsHandle(bool ownsHandle) : base(ownsHandle)
    {
        
    }

    public GpuClonerOptionsHandle(IntPtr handle) : base(true)
    {
        SetHandle(handle);
    }

    protected override bool ReleaseHandle()
    {
        GpuNative.faiss_GpuClonerOptions_free(handle);
        return true;
    }
}