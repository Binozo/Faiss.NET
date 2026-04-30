using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

/// <summary>
/// Exact binary flat index. Performs exhaustive Hamming search on packed binary vectors.
/// </summary>
public sealed class IndexBinaryFlat : CpuBinaryIndex<IndexBinaryFlat>, IFromNativeBinaryHandle<IndexBinaryFlat>
{
    /// <summary>
    /// Creates an exact binary flat index.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality in bits.</param>
    public IndexBinaryFlat(int dimensions)
    {
        if (dimensions == 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be dividable by 8", nameof(dimensions));
        }

        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, "BFlat")
        );

        SafeHandle = new FaissIndexBinaryHandle(ptr);
    }

    private IndexBinaryFlat(IntPtr handle) : base(handle)
    {
    }

    static IndexBinaryFlat IFromNativeBinaryHandle<IndexBinaryFlat>.FromHandle(IntPtr handle) => new(handle);
}