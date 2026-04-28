using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Mapped;

/// <summary>
/// Wraps any index to add support for custom vector IDs with reverse lookup capability.
/// Extends <see cref="IndexIDMap{T}"/> by maintaining a reverse map from custom IDs
/// to internal sequential IDs, enabling efficient vector reconstruction by custom ID.
/// </summary>
/// <typeparam name="T">The type of the underlying index to wrap.</typeparam>
/// <remarks>
/// Use this wrapper with indexes that do not natively support custom IDs
/// (e.g. <see cref="IndexFlatL2"/>, <see cref="IndexHNSW"/>, <see cref="IndexPQ"/>).
/// Only <see cref="Add(long, ReadOnlySpan{float}, ReadOnlySpan{long})"/> is supported;
/// <see cref="Index{T}.Add(long, ReadOnlySpan{float})"/> will throw.
/// The reverse map is automatically maintained during <see cref="Add"/> and
/// is rebuilt when constructing from an existing native handle.
/// </remarks>
public class IndexIDMap2<T> : CpuIndex<IndexIDMap2<T>>, IFromNativeHandle<IndexIDMap2<T>>, IIndexIDMapped where T : CpuIndex<T>, IFromNativeHandle<T>, ISequentialIDIndex
{
    private readonly T _index;

    public IndexIDMap2(T index, bool takeOwnership = false)
    {
        _index = index;

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIDMap2_new(out IntPtr ptr, index.Handle)
        );

        SafeHandle = new FaissIndexHandle(ptr);

        Native.faiss_IndexIDMap2_set_own_fields(SafeHandle, takeOwnership);
        if (takeOwnership)
        {
            _index.Handle.SetHandleAsInvalid();
        }
    }

    private IndexIDMap2(IntPtr handle, bool takeOwnership = false) : base(handle)
    {
        _index = T.FromHandle(Native.faiss_IndexIDMap2_sub_index(new FaissIndexHandle(handle)));
        ReconstructRevMap();
        

        Native.faiss_IndexIDMap2_set_own_fields(SafeHandle, takeOwnership);
        if (takeOwnership)
        {
            _index.Handle.SetHandleAsInvalid();
        }
    }

    static IndexIDMap2<T> IFromNativeHandle<IndexIDMap2<T>>.FromHandle(IntPtr handle) => new(handle);

    private void ReconstructRevMap()
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIDMap2_construct_rev_map(SafeHandle)
        );
    }
    
    public unsafe void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (xids.Length < count)
        {
            throw new ArgumentException("Not enough custom IDs for the vectors.", nameof(xids));
        }

        fixed (float* pVectors = vectors)
        fixed (long* pXids = xids)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_add_with_ids(SafeHandle, count, pVectors, pXids)
            );
        }
    }
}