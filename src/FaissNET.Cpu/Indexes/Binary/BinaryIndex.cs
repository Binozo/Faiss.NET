using Faiss.Cpu.Exceptions;
using Faiss.Cpu.Interfaces;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Binary;

public abstract class BinaryIndex<T> : INativeBinaryIndex<T>, ITrainableBinaryIndex where T : BinaryIndex<T>
{
    FaissIndexBinaryHandle INativeBinaryIndex.Handle => SafeHandle;

    private protected FaissIndexBinaryHandle SafeHandle { get; init; }

    private protected BinaryIndex()
    {
    }

    private protected BinaryIndex(IntPtr handle)
    {
        SafeHandle = new FaissIndexBinaryHandle(handle);
    }

    public int Dimensions => Native.faiss_IndexBinary_d(SafeHandle);

    public long TotalCount => Native.faiss_IndexBinary_ntotal(SafeHandle);

    public bool IsTrained => Native.faiss_IndexBinary_is_trained(SafeHandle) != 0;

    public MetricType Metric => Native.faiss_IndexBinary_metric_type(SafeHandle);

    public unsafe void Add(long count, ReadOnlySpan<byte> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        fixed (byte* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinary_add(SafeHandle, count, pVectors));
        }
    }

    public unsafe void Search(long count, ReadOnlySpan<byte> queryVectors, int k, Span<int> distances, Span<long> labels)
    {
        fixed (byte* pQuery = queryVectors)
        fixed (int* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_search(SafeHandle, count, pQuery, k, pDistances, pLabels)
            );
        }
    }

    public unsafe void Assign(long count, ReadOnlySpan<byte> queryVectors, long k, Span<long> labels)
    {
        fixed (byte* pQuery = queryVectors)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_assign(SafeHandle, count, pQuery, pLabels, k)
            );
        }
    }

    public unsafe void SearchWithParams(
        long count,
        ReadOnlySpan<byte> queryVectors,
        int k,
        ISearchParameters parameters,
        Span<int> distances,
        Span<long> labels)
    {
        fixed (byte* pQuery = queryVectors)
        fixed (int* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_search_with_params(
                    SafeHandle, count, pQuery, k, ((INativeSearchParameters)parameters).DangerousGetHandle(), pDistances, pLabels)
            );
        }
    }
    
    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors)
    {
        return Task.Run(() =>
        {
            unsafe
            {
                using var handle = vectors.Pin();
                byte* pVectors = (byte*)handle.Pointer;

                FaissErrorHandler.ThrowIfError(
                    Native.faiss_IndexBinary_train(SafeHandle, count, pVectors)
                );
            }
        });
    }

    public void Reset()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinary_reset(SafeHandle));
    }

    public virtual void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}