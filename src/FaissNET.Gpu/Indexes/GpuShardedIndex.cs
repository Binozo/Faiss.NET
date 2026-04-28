using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Interfaces;

namespace Faiss.Gpu.Indexes;

public sealed class GpuShardedIndex<T> : Index<T>, INativeGpuIndex<T> where T : Index<T>, INativeIndex<T>, IFromNativeHandle<T>
{
    public readonly int[] Devices;
    
    internal GpuShardedIndex(IntPtr handle, int[] devices) : base(handle)
    {
        Devices = devices;
    }
}