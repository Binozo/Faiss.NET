using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Binary;

/// <inheritdoc cref="IBinaryIndex" />
public abstract class BinaryIndex : IBinaryIndex, INativeBinaryIndex
{
    private readonly FaissBinaryIndexHandle _handle;

    protected BinaryIndex(FaissBinaryIndexHandle handle) => _handle = handle ?? throw new ArgumentNullException(nameof(handle));

    protected internal FaissBinaryIndexHandle NativeHandle => _handle;

    FaissBinaryIndexHandle INativeBinaryIndex.Handle => _handle;

    public int Dimensions => Native.faiss_IndexBinary_d(NativeHandle);

    public long TotalCount => Native.faiss_IndexBinary_ntotal(NativeHandle);

    public MetricType Metric => Native.faiss_IndexBinary_metric_type(NativeHandle);

    public unsafe void Search(long count, ReadOnlySpan<byte> queryVectors, int k, Span<int> distances, Span<long> labels)
    {
        fixed (byte* pQuery = queryVectors)
        fixed (int* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_search(NativeHandle, count, pQuery, k, pDistances, pLabels)
            );
        }
    }

    public unsafe void Assign(long count, ReadOnlySpan<byte> queryVectors, long k, Span<long> labels)
    {
        fixed (byte* pQuery = queryVectors)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_assign(NativeHandle, count, pQuery, pLabels, k)
            );
        }
    }

    public void Reset() => FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinary_reset(NativeHandle));

    public virtual void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}