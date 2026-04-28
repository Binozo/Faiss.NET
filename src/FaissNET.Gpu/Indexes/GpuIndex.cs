using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Interfaces;

namespace Faiss.Gpu.Indexes;

public sealed class GpuIndex<T> : Index<T>, INativeGpuIndex<T> where T : Index<T>, INativeIndex<T>, IFromNativeHandle<T>
{
    public readonly int Device;
    
    internal GpuIndex(IntPtr handle, int device) : base(handle)
    {
        Device = device;
    }
}