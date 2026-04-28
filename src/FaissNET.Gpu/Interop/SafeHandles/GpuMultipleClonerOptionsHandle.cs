using Faiss.Gpu.Interop.NativeMethods;

namespace Faiss.Gpu.Interop.SafeHandles;

internal sealed class GpuMultipleClonerOptionsHandle : GpuClonerOptionsHandle
{
    public GpuMultipleClonerOptionsHandle(IntPtr handle) : base(true)
    {
        SetHandle(handle);
    }

    protected override bool ReleaseHandle()
    {
        GpuNative.faiss_GpuMultipleClonerOptions_free(handle);
        return true;
    }
}