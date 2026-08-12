using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes;

/// <inheritdoc cref="IFloatIndex" />
public abstract class FloatIndex : IFloatIndex, INativeIndex
{
    private readonly FaissIndexHandle _handle;
    
    protected FloatIndex(FaissIndexHandle handle) => _handle = handle ?? throw new ArgumentNullException(nameof(handle));

    protected internal FaissIndexHandle NativeHandle => _handle;

    FaissIndexHandle INativeIndex.Handle => _handle;

    public int Dimensions => Native.faiss_Index_d(NativeHandle);
    public long TotalCount => Native.faiss_Index_ntotal(NativeHandle);

    public MetricType Metric => Native.faiss_Index_metric_type(NativeHandle);

    public virtual unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(NativeHandle, count, pQuery, k, pDistances, pLabels));
        }
    }

    public void Assign(long count, ReadOnlySpan<float> queryVectors, long k, Span<long> labels)
    {
        unsafe
        {
            fixed (float* pQuery = queryVectors)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_assign(NativeHandle, count, pQuery, pLabels, k)
                );
            }
        }
    }

    public void Reset() => FaissErrorHandler.ThrowIfError(Native.faiss_Index_reset(NativeHandle));


    public virtual void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}