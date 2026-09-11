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
    
    internal FloatIndex(FaissIndexHandle handle) => _handle = handle ?? throw new ArgumentNullException(nameof(handle));

    internal FaissIndexHandle NativeHandle => _handle;

    FaissIndexHandle INativeIndex.Handle => _handle;

    public int Dimensions => Native.faiss_Index_d(NativeHandle);
    public long TotalCount => Native.faiss_Index_ntotal(NativeHandle);

    public MetricType Metric => Native.faiss_Index_metric_type(NativeHandle);

    public virtual unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

        long queryLength = checked(count * Dimensions);
        long resultLength = checked(count * k);

        if (queryVectors.Length < queryLength)
            throw new ArgumentException($"Query span too small. Expected {queryLength}, got {queryVectors.Length}.", nameof(queryVectors));

        if (distances.Length < resultLength)
            throw new ArgumentException($"Distance span too small. Expected {resultLength}, got {distances.Length}.", nameof(distances));

        if (labels.Length < resultLength)
            throw new ArgumentException($"Label span too small. Expected {resultLength}, got {labels.Length}.", nameof(labels));

        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(NativeHandle, count, pQuery, k, pDistances, pLabels));
        }
    }

    public void Assign(long count, ReadOnlySpan<float> queryVectors, long k, Span<long> labels)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

        long queryLength = checked(count * Dimensions);
        long resultLength = checked(count * k);

        if (queryVectors.Length < queryLength)
            throw new ArgumentException($"Query span too small. Expected {queryLength}, got {queryVectors.Length}.", nameof(queryVectors));

        if (labels.Length < resultLength)
            throw new ArgumentException($"Label span too small. Expected {resultLength}, got {labels.Length}.", nameof(labels));

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
    
    public bool Verbose
    {
        get => Native.faiss_Index_verbose(NativeHandle) != 0;
        set => Native.faiss_Index_set_verbose(NativeHandle, value);
    }


    public virtual void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}