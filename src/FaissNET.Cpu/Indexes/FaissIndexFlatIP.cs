namespace Faiss.Cpu.Indexes;

using Interfaces;
using Exceptions;
using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;

/// <summary>
/// Exact search for Inner Product (useful for Cosine Similarity).
/// Ideal for NLP and embedding-based search.
/// </summary>
public sealed class FaissIndexFlatIP : FaissCpuIndex, INativeFaissCpuIndex
{
    private readonly FaissIndexHandle _handle;
    private protected override FaissIndexHandle NativeHandle => _handle;

    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public FaissIndexFlatIP(int dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatIP_new_with(out _handle, dimensions));
    }

    internal FaissIndexFlatIP(IntPtr handle)
    {
        _handle = new FaissIndexHandle(handle);
    }

    static INativeFaissIndex INativeFaissCpuIndex.FromHandle(IntPtr handle) => new FaissIndexFlatIP(handle);
}