using Faiss.Cpu.Exceptions;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes;

/// <inheritdoc cref="IIndex" />
public abstract class Index<T> : INativeIndex<T>, ITrainableIndex where T : Index<T>
{
    FaissIndexHandle INativeIndex.Handle => SafeHandle;

    protected internal FaissIndexHandle SafeHandle { get; protected init; }

    private protected Index()
    {
        
    }

    private protected Index(IntPtr handle)
    {
        SafeHandle = new FaissIndexHandle(handle);
    }

    public int Dimensions => Native.faiss_Index_d(SafeHandle);
    public long TotalCount => Native.faiss_Index_ntotal(SafeHandle);
    public bool IsTrained => Native.faiss_Index_is_trained(SafeHandle) != 0;

    public MetricType Metric => Native.faiss_Index_metric_type(SafeHandle);

    public unsafe void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(SafeHandle, count, pVectors));
        }
    }

    public unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(SafeHandle, count, pQuery, k, pDistances, pLabels));
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
                    Native.faiss_Index_assign(SafeHandle, count, pQuery, pLabels, k)
                );
            }
        }
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels)
    {
        unsafe
        {
            fixed (float* pQuery = queryVectors)
            fixed (float* pDistances = distances)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_search_with_params(SafeHandle, count, pQuery, k, ((INativeSearchParameters)parameters).DangerousGetHandle(), pDistances, pLabels)
                );
            }
        }
    }
    
    public unsafe void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result)
    {
        fixed (float* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_range_search(SafeHandle, count, pQuery, radius, result.SafeHandle)
            );
        }
    }
    
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors)
    {
        return Task.Run(() =>
        {
            unsafe
            {
                using var handle = vectors.Pin();
                float* pVectors = (float*)handle.Pointer;

                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_train(SafeHandle, count, pVectors)
                );
            }
        });
    }

    public void Reset()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_Index_reset(SafeHandle));
    }


    public virtual void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}