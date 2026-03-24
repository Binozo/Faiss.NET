namespace Faiss.Cpu;

using Models;
using Interfaces;
using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;

public abstract class FaissCpuIndex : INativeFaissIndex
{
    private protected abstract FaissIndexHandle NativeHandle { get; }

    public IntPtr Handle => NativeHandle.DangerousGetHandle();

    public int Dimensions => Native.faiss_Index_d(NativeHandle);

    public long TotalCount => Native.faiss_Index_ntotal(NativeHandle);

    public bool IsTrained => Native.faiss_Index_is_trained(NativeHandle) != 0;

    public MetricType Metric => Native.faiss_Index_metric_type(NativeHandle);

    public unsafe void Add(long count, ReadOnlySpan<float> vectors)
    {
        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(NativeHandle, count, pVectors));
        }
    }

    public unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(NativeHandle, count, pQuery, k, pDistances, pLabels));
        }
    }

    public void Reset()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_Index_reset(NativeHandle));
    }

    public void Dispose()
    {
        NativeHandle.Dispose();
    }
}