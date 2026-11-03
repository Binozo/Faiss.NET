using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Interop.NativeMethods;

namespace Faiss.Gpu.Indexes;

public static class GpuIndexExtensions
{
    extension(IGpuIndex gpuIndex)
    {
        public int DeviceId => GpuNative.faiss_GpuIndex_getDevice(gpuIndex.Handle);
    }
}