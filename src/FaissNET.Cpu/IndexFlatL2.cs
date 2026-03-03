using Faiss.Interfaces;
using Faiss.Interop.ErrorHandling;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu;

/// <summary>
/// Exact search for L2 (Euclidean) distance.
/// The most basic and accurate Faiss index.
/// </summary>
public sealed class IndexFlatL2 : IFaissIndex
{
    private readonly FaissIndexHandle _handle;

    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatL2(int dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatL2_new_with(out _handle, dimensions));
    }

    public int Dimensions => Native.faiss_Index_d(_handle);
    
    public long TotalCount => Native.faiss_Index_ntotal(_handle);
    
    // Flat indexes don't require training, but we map it anyway for the interface
    public bool IsTrained => Native.faiss_Index_is_trained(_handle) != 0;

    public unsafe void Add(long count, ReadOnlySpan<float> vectors)
    {
        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(_handle, count, pVectors));
        }
    }

    public unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(_handle, count, pQuery, k, pDistances, pLabels));
        }
    }

    public void Reset()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_Index_reset(_handle));
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}