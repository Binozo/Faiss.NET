using Faiss.Exceptions;
using Faiss.Interop.Errors;

namespace Faiss.Cpu.Indexes;

using Interfaces;
using Interop.NativeMethods;

/// <summary>
/// Exact search for Inner Product (useful for Cosine Similarity).
/// Ideal for NLP and embedding-based search.
/// </summary>
public sealed class IndexFlatIP : CpuIndex<IndexFlatIP>, IFromNativeHandle<IndexFlatIP>, IFlatIndex, ISequentialIDIndex
{
    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatIP(long dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatIP_new_with(out var handle, dimensions));

        SafeHandle = handle;
    }

    private IndexFlatIP(IntPtr handle) : base(handle)
    {

    }

    static IndexFlatIP IFromNativeHandle<IndexFlatIP>.FromHandle(IntPtr handle) => new(handle);
}
