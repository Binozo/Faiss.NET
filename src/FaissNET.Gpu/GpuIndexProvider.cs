namespace Faiss.Gpu;

using Faiss.Cpu.Interfaces;

using Interfaces;
using Faiss.Interop.ErrorHandling;
using Interop;

public static class GpuIndexProvider
{
    /// <summary>
    /// Moves a CPU index to the specified GPU device using the provided VRAM context.
    /// </summary>
    public static INativeFaissGpuIndex<T> TransferToGpu<T>(FaissGpuContext context, T cpuIndex, int deviceId = 0) where T : INativeFaissCpuIndex
    {
        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_index_cpu_to_gpu(
            context.ResourcePointer,
            deviceId,
            cpuIndex.Handle,
            out IntPtr gpuIndexPtr));

        return new GpuFaissIndexWrapper<T>(gpuIndexPtr);
    }
    
    /// <summary>
    /// Moves the index out of VRAM and puts it back into RAM.
    /// </summary>
    public static T TransferToCpu<T>(INativeFaissGpuIndex<T> gpuIndex) where T : INativeFaissCpuIndex
    {
        FaissErrorHandler.ThrowIfError(GpuNativeMethods.faiss_index_gpu_to_cpu(
            gpuIndex.Handle,
            out IntPtr cpuIndexPtr));

        return (T)T.FromHandle(cpuIndexPtr);
    }
}