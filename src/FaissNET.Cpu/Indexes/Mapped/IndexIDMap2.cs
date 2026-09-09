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
public sealed class IndexIDMap2<T> : MappedIndex<IndexIDMap2<T>, T>, IFromNativeIndexHandle<IndexIDMap2<T>>, IComputeResidualFloatIndex, IReconstructFloatIndex where T : class, IIDSequentialFloatIndex, IFloatIndex, IFromNativeIndexHandle<T>
{
    private readonly T _subIndex;
    private readonly bool _takeOwnership;

    public IndexIDMap2(T index, bool takeOwnership = false) : this(index.Handle, index, takeOwnership)
    {
    }

    private IndexIDMap2(FaissIndexHandle subIndexHandle, T? subIndex = null, bool takeOwnership = false) : base(CreateHandle(subIndexHandle))
    {
        _subIndex = subIndex ?? T.FromPointer(Native.faiss_IndexIDMap2_sub_index(subIndexHandle));
        OwnsSubIndex = takeOwnership;
        _takeOwnership = takeOwnership;
        
        ReconstructRevMap();
    }

    public bool OwnsSubIndex
    {
        get => Native.faiss_IndexIDMap2_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIDMap2_set_own_fields(NativeHandle, value);
    }

    private void ReconstructRevMap() => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap2_construct_rev_map(NativeHandle));

    public float[] Reconstruct(long key) =>  ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count)  => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    private static FaissIndexHandle CreateHandle(FaissIndexHandle subIndexHandle)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIDMap2_new(out var ptr, subIndexHandle));
        return new FaissIndexHandle(ptr);
    }

    static IndexIDMap2<T> IFromNativeIndexHandle<IndexIDMap2<T>>.FromHandle(FaissIndexHandle handle) => new(handle, takeOwnership: true);

    public override void Dispose()
    {
        if (_takeOwnership)
        {
            _subIndex.Handle.SetHandleAsInvalid();
        }

        base.Dispose();
    }
}