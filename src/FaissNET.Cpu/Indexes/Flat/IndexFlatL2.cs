using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

using Interfaces;
using Interop.NativeMethods;

internal readonly struct IndexFlatL2Release : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexFlatL2_free(handle);
}

/// <summary>
/// Exact search for L2 (Euclidean) distance.
/// The most basic and accurate Faiss index.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public sealed class IndexFlatL2 : CpuFlatFloatIndex<IndexFlatL2>, IFromNativeIndexHandle<IndexFlatL2>, IGpuClonableIndex<IndexFlatL2, GpuIndexFlatL2>
{
    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatL2(long dimensions) : this(CreateHandle(dimensions))
    {
    }

    private IndexFlatL2(FaissIndexHandle handle) : base(handle)
    {
    }
    
    private static FaissIndexHandle CreateHandle(long dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatL2_new_with(out var handle, dimensions));
        return new FaissIndexHandle<IndexFlatL2Release>(handle);
    }

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexFlatL2Release>(handle, ownsHandle);

    static IndexFlatL2 IFromNativeIndexHandle<IndexFlatL2>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexFlatL2 IFromNativeIndexHandle<IndexFlatL2>.FromHandle(FaissIndexHandle handle) => new(handle);
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexFlatL2 : GpuFlatFloatIndex<GpuIndexFlatL2>, IFromNativeIndexHandle<GpuIndexFlatL2>, IGpuIndex<IndexFlatL2>
{
    private GpuIndexFlatL2(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexFlatL2 IFromNativeIndexHandle<GpuIndexFlatL2>.FromHandle(FaissIndexHandle handle) => new(handle);
}