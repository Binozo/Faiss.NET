namespace Faiss.Gpu;

using Cpu;
using Interfaces;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

public sealed class GpuFaissIndexWrapper<T> : FaissCpuIndex, INativeFaissGpuIndex<T> where T : INativeFaissCpuIndex
{
    private protected override FaissIndexHandle NativeHandle { get; }

    internal GpuFaissIndexWrapper(IntPtr ptr)
    {
        NativeHandle = new FaissIndexHandle(ptr);
    }
}