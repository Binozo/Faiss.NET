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
    private readonly T _subIndex;
    private readonly bool _takeOwnership;

    public IndexIDMap(T index, bool takeOwnership = false) : this(index.Handle, index, takeOwnership)
    {
    }

    private IndexIDMap(FaissIndexHandle subIndexHandle, T? subIndex = null, bool takeOwnership = false) : base(CreateHandle(subIndexHandle))
    {
        _subIndex = subIndex ?? T.FromPointer(Native.faiss_IndexIDMap_sub_index(subIndexHandle));
        OwnsSubIndex = takeOwnership;
        _takeOwnership = takeOwnership;
    }

    public bool OwnsSubIndex
    {
        get => Native.faiss_IndexIDMap_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIDMap_set_own_fields(NativeHandle, value);
    }

    private static FaissIndexHandle CreateHandle(FaissIndexHandle subIndexHandle)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap_new(out var ptr, subIndexHandle));
        return new FaissIndexHandle(ptr);
    }

    static IndexIDMap<T> IFromNativeIndexHandle<IndexIDMap<T>>.FromHandle(FaissIndexHandle handle) => new(handle, takeOwnership: true);

    public override void Dispose()
    {
        if (_takeOwnership)
        {
            _subIndex.Handle.SetHandleAsInvalid();
        }

        base.Dispose();
    }
}