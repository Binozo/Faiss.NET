using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Mapped;

/// <summary>
/// Wraps any unpopulated index to add support for custom vector IDs.
/// Translates internal sequential IDs to user-provided IDs on search results.
/// </summary>
/// <typeparam name="T">The type of the underlying index to wrap.</typeparam>
/// <remarks>
/// Use this wrapper with indexes that do not natively support custom IDs
/// (e.g. <see cref="IndexFlatL2"/>, <see cref="IndexHNSW"/>, <see cref="IndexPQ"/>).
/// </remarks>
public sealed class IndexIDMap<T> : MappedIndex<IndexIDMap<T>, T>, IFromNativeIndexHandle<IndexIDMap<T>> where T : class, IIDSequentialFloatIndex, IFloatIndex, IFromNativeIndexHandle<T>
{
    public readonly T SubIndex;

    /// <param name="index">The index to wrap. Must be empty.</param>
    /// <param name="takeOwnership">
    /// If <c>true</c> the map frees <paramref name="index"/> and <paramref name="index"/> is closed immediately: use <see cref="SubIndex"/> instead.
    /// If <c>false</c> the caller keeps ownership and keeps using <paramref name="index"/>.
    /// </param>
    public IndexIDMap(T index, bool takeOwnership = false) : base(CreateHandle(OwnershipGuard.RequireIndependent(index, nameof(index)).Handle, takeOwnership))
    {
        OwnsSubIndex = takeOwnership;

        if (takeOwnership)
        {
            index.Handle.SetHandleAsInvalid();
            SubIndex = T.FromBorrowed(Native.faiss_IndexIDMap_sub_index(NativeHandle), NativeHandle);
        }
        else
        {
            SubIndex = index;
        }
    }

    private IndexIDMap(FaissIndexHandle handle) : base(handle)
    {
        // Freeing is being taken care of by faiss' default behavior.
        SubIndex = T.FromBorrowed(Native.faiss_IndexIDMap_sub_index(handle), handle);
    }

    public bool OwnsSubIndex
    {
        get => Native.faiss_IndexIDMap_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIDMap_set_own_fields(NativeHandle, value);
    }

    private static FaissIndexHandle CreateHandle(FaissIndexHandle subIndexHandle, bool takeOwnership)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap_new(out var ptr, subIndexHandle));

        if (takeOwnership)
            return new FaissIndexHandle(ptr);

        var handle = new FaissDependentIndexHandle<IndexRelease>(ptr);
        handle.AddDependency(subIndexHandle);
        return handle;
    }

    static IndexIDMap<T> IFromNativeIndexHandle<IndexIDMap<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public override void Dispose()
    {
        // A view of the sub-index references this map's own handle and keeps it alive thus it must be released first
        if (SubIndex.Handle is FaissBorrowedIndexHandle)
        {
            SubIndex.Dispose();
        }

        base.Dispose();
    }
}