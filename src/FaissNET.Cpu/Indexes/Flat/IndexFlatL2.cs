using Faiss.Exceptions;
using Faiss.Interop.Errors;

namespace Faiss.Cpu.Indexes.Flat;

using Interfaces;
using Interop.NativeMethods;

/// <summary>
/// Exact search for L2 (Euclidean) distance.
/// The most basic and accurate Faiss index.
/// </summary>
public sealed class IndexFlatL2 : CpuIndex<IndexFlatL2>, IFromNativeHandle<IndexFlatL2>, IFlatIndex
{
    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatL2(long dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatL2_new_with(out var handle, dimensions));

        SafeHandle = handle;
    }

    private IndexFlatL2(IntPtr handle) : base(handle)
    {
    }

    static IndexFlatL2 IFromNativeHandle<IndexFlatL2>.FromHandle(IntPtr handle) => new(handle);
}