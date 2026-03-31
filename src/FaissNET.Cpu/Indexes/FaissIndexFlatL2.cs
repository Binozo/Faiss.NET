namespace Faiss.Cpu.Indexes;

using Interfaces;
using Exceptions;
using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;


/// <summary>
/// Exact search for L2 (Euclidean) distance.
/// The most basic and accurate Faiss index.
/// </summary>
public sealed class FaissIndexFlatL2 : FaissCpuIndex, INativeFaissCpuIndex
{
    private readonly FaissIndexHandle _handle;
    private protected override FaissIndexHandle NativeHandle => _handle;

    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public FaissIndexFlatL2(long dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatL2_new_with(out _handle, dimensions));
    }

    internal FaissIndexFlatL2(IntPtr handle)
    {
        _handle = new FaissIndexHandle(handle);
    }

    static INativeFaissIndex INativeFaissCpuIndex.FromHandle(IntPtr handle) => new FaissIndexFlatL2(handle);
}