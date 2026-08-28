using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

using Interfaces;
using Interop.NativeMethods;

internal readonly struct IndexFlatIPRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexFlatIP_free(handle);
}

/// <summary>
/// Exact search for Inner Product (useful for Cosine Similarity).
/// Ideal for NLP and embedding-based search.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public class IndexFlatIP : CpuFlatFloatIndex<IndexFlatIP>, IFromNativeIndexHandle<IndexFlatIP>, ISerializableFloatIndex, IGpuClonableIndex<IndexFlatIP, GpuIndexFlatIP>
{
    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexFlatIP(long dimensions) : this(CreateHandle(dimensions))
    {
    }

    private IndexFlatIP(FaissIndexHandle handle) : base(handle)
    {
    }

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexFlatIPRelease>(handle, ownsHandle);

    private static FaissIndexHandle CreateHandle(long dimensions)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlatIP_new_with(out var ptr, dimensions));
        return new FaissIndexHandle<IndexFlatIPRelease>(ptr);
    }
    
    static IndexFlatIP IFromNativeIndexHandle<IndexFlatIP>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexFlatIP IFromNativeIndexHandle<IndexFlatIP>.FromHandle(FaissIndexHandle handle) => new(handle);
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexFlatIP : GpuFlatFloatIndex<GpuIndexFlatIP>, IFromNativeIndexHandle<GpuIndexFlatIP>, IGpuIndex<IndexFlatIP>
{
    private GpuIndexFlatIP(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexFlatIP IFromNativeIndexHandle<GpuIndexFlatIP>.FromHandle(FaissIndexHandle handle) => new(handle);
}