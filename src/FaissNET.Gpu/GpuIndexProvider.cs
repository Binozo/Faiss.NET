using Faiss.Cpu;

namespace Faiss.Gpu;

using Interfaces;
using Faiss.Interop.ErrorHandling;
using Interop;

public static class GpuIndexProvider
{
    /// <summary>
    /// Moves a CPU index to the specified GPU device using the provided VRAM context.
    /// </summary>
    public static IFaissIndex TransferToGpu(FaissGpuContext context, IFaissIndex cpuIndex, int deviceId = 0)
    {
        if (cpuIndex is not INativeIndex nativeCpuIndex)
        {
            throw new ArgumentException("This index doesn't support native pointer extraction.", nameof(cpuIndex));
        }

        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_index_cpu_to_gpu(
            context.ResourcePointer,
            deviceId,
            nativeCpuIndex.Handle,
            out IntPtr gpuIndexPtr));

        return new GpuIndexWrapper(gpuIndexPtr);
    }
    
    /// <summary>
    /// Moves the index out of VRAM and puts it back into RAM.
    /// </summary>
    public static IFaissIndex TransferToCpu(IFaissIndex gpuIndex)
    {
        if (gpuIndex is not INativeIndex nativeGpuIndex)
        {
            throw new ArgumentException("This isn't a native index.", nameof(gpuIndex));
        }

        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_index_gpu_to_cpu(
            nativeGpuIndex.Handle,
            out IntPtr cpuIndexPtr));

        return new GenericCpuIndexWrapper(cpuIndexPtr);
    }
}