using Faiss.Interfaces;
using Faiss.Interop.ErrorHandling;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu;

/// <summary>
/// Exact search for Inner Product (useful for Cosine Similarity).
/// Ideal for NLP and embedding-based search.
/// </summary>
public sealed class IndexFlatIP : IFaissIndex, INativeIndex
{
    private readonly FaissIndexHandle _handle;
    
    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatIP(int dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatIP_new_with(out _handle, dimensions));
    }
    
    public IntPtr Handle => _handle.DangerousGetHandle();

    public int Dimensions => Native.faiss_Index_d(_handle);
    
    public long TotalCount => Native.faiss_Index_ntotal(_handle);
    
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