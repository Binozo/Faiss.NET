using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Mapped;

/// <summary>
/// Wraps any unpopulated index to add support for custom vector IDs with reverse lookup capability.
/// Extends <see cref="IndexIDMap{T}"/> by maintaining a reverse map from custom IDs
/// to internal sequential IDs, enabling efficient vector reconstruction by custom ID.
/// </summary>
/// <typeparam name="T">The type of the underlying index to wrap.</typeparam>
/// <remarks>
/// Use this wrapper with indexes that do not natively support custom IDs
/// (e.g. <see cref="IndexFlatL2"/>, <see cref="IndexHNSW"/>, <see cref="IndexPQ"/>).
/// </remarks>
public sealed class IndexIDMap2<T> : MappedIndex<IndexIDMap2<T>, T>, IFromNativeIndexHandle<IndexIDMap2<T>>, IComputeResidualFloatIndex, IReconstructFloatIndex, IKeyLayout where T : class, IIDSequentialFloatIndex, IFloatIndex, IFromNativeIndexHandle<T>
{
    /// <inheritdoc cref="IndexIDMap{T}.SubIndex"/>
    public readonly T SubIndex;

    /// <inheritdoc cref="IndexIDMap{T}(T, bool)"/>
    public IndexIDMap2(T index, bool takeOwnership = false) : base(CreateHandle(OwnershipGuard.RequireIndependent(index, nameof(index)).Handle, takeOwnership))
    {
        OwnsSubIndex = takeOwnership;

        if (takeOwnership)
        {
            index.Handle.SetHandleAsInvalid();
            SubIndex = T.FromBorrowed(Native.faiss_IndexIDMap2_sub_index(NativeHandle), NativeHandle);
        }
        else
        {
            SubIndex = index;
        }

        ReconstructRevMap();
    }

    private IndexIDMap2(FaissIndexHandle handle) : base(handle)
    {
        SubIndex = T.FromBorrowed(Native.faiss_IndexIDMap2_sub_index(handle), handle);

        ReconstructRevMap();
    }

    public bool OwnsSubIndex
    {
        get => Native.faiss_IndexIDMap2_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIDMap2_set_own_fields(NativeHandle, value);
    }

    private void ReconstructRevMap() => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap2_construct_rev_map(NativeHandle));

    KeyLayout IKeyLayout.KeyLayout => KeyLayout.Ids;

    public float[] Reconstruct(long key) =>  ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count)  => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    private static FaissIndexHandle CreateHandle(FaissIndexHandle subIndexHandle, bool takeOwnership)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap2_new(out var ptr, subIndexHandle));

        if (takeOwnership)
            return new FaissIndexHandle(ptr);

        var handle = new FaissDependentIndexHandle<IndexRelease>(ptr);
        handle.AddDependency(subIndexHandle);
        return handle;
    }

    static IndexIDMap2<T> IFromNativeIndexHandle<IndexIDMap2<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

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