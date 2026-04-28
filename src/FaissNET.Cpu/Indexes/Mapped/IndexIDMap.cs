using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Mapped;

/// <summary>
/// Wraps any index to add support for custom vector IDs.
/// Translates internal sequential IDs to user-provided IDs on search results.
/// </summary>
/// <typeparam name="T">The type of the underlying index to wrap.</typeparam>
/// <remarks>
/// Use this wrapper with indexes that do not natively support custom IDs
/// (e.g. <see cref="IndexFlatL2"/>, <see cref="IndexHNSW"/>, <see cref="IndexPQ"/>).
/// Only <see cref="Add(long, ReadOnlySpan{float}, ReadOnlySpan{long})"/> is supported;
/// <see cref="Index{T}.Add(long, ReadOnlySpan{float})"/> will throw.
/// </remarks>
public class IndexIDMap<T> : CpuIndex<IndexIDMap<T>>, IFromNativeHandle<IndexIDMap<T>>, IIndexIDMapped where T : CpuIndex<T>, IFromNativeHandle<T>, ISequentialIDIndex
{
    private readonly T _index;

    public IndexIDMap(T index, bool takeOwnership = false)
    {
        _index = index;

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIDMap_new(out IntPtr ptr, index.Handle)
        );

        SafeHandle = new FaissIndexHandle(ptr);

        Native.faiss_IndexIDMap_set_own_fields(SafeHandle, takeOwnership);
        if (takeOwnership)
        {
            _index.Handle.SetHandleAsInvalid();
        }
    }

    private IndexIDMap(IntPtr handle, bool takeOwnership = false) : base(handle)
    {
        _index = T.FromHandle(Native.faiss_IndexIDMap_sub_index(new FaissIndexHandle(handle)));
        

        Native.faiss_IndexIDMap_set_own_fields(SafeHandle, takeOwnership);
        if (takeOwnership)
        {
            _index.Handle.SetHandleAsInvalid();
        }
    }

    static IndexIDMap<T> IFromNativeHandle<IndexIDMap<T>>.FromHandle(IntPtr handle) => new(handle);
    
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