using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Refine;

internal readonly struct IndexRefineFlatRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexRefineFlat_free(handle);
}

/// <summary>
/// Wraps a fast approximate index and re-ranks results using an exact flat index.
/// </summary>
public sealed class IndexRefineFlat<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IRangeSearchFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IFromNativeIndexHandle<IndexRefineFlat<T>>, IClonableFloatIndex<IndexRefineFlat<T>> where T : FloatIndex, IIDSequentialFloatIndex, IFromNativeIndexHandle<T>
{
    public readonly T BaseIndex;

    /// <param name="baseIndex">The empty approximate index to refine.</param>
    /// <param name="ownBaseIndex">
    /// If <c>true</c>, this index frees the quantizer and <paramref name="baseIndex"/> is closed immediately.
    /// If <c>false</c>, the caller keeps ownership. This index keeps the quantizer alive until it is freed, even if the caller disposes it first.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="baseIndex"/> is part of another index, such as <c>IndexIDMap.SubIndex</c>.</exception>
    public IndexRefineFlat(T baseIndex, bool ownBaseIndex = true)
        : this(CreateHandle(OwnershipGuard.RequireIndependent(baseIndex, nameof(baseIndex)), ownBaseIndex), ownBaseIndex, baseIndex)
    {
        if (ownBaseIndex)
        {
            baseIndex.Handle.SetHandleAsInvalid();
        }
    }

    /// <param name="ownBaseIndex">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    internal IndexRefineFlat(FaissIndexHandle handle, bool? ownBaseIndex = null, T? baseIndex = null) : base(handle)
    {
        if (ownBaseIndex is bool own)
        {
            OwnBaseIndex = own;
        }

        BaseIndex = ownBaseIndex switch
        {
            // A view of the caller's index, created while the caller's handle is still open.
            true => T.FromBorrowed(baseIndex!.Handle.DangerousGetHandle(), handle),
            false => baseIndex!,
            null => T.FromBorrowed(Native.faiss_IndexRefineFlat_base_index(handle), handle),
        };
    }

    public bool OwnBaseIndex
    {
        get => Native.faiss_IndexRefineFlat_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexRefineFlat_set_own_fields(NativeHandle, value);
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    /// <summary>
    /// Oversampling factor for refinement, in [1, 1000].
    /// When searching for k results, the base index is queried for k * KFactor candidates,
    /// which are then re-ranked exactly by the internal flat index.
    /// </summary>
    public float KFactor
    {
        get => Native.faiss_IndexRefineFlat_k_factor(NativeHandle);
        set
        {
            if (!float.IsFinite(value) || value < 1.0f || value > 1000.0f)
                throw new ArgumentOutOfRangeException(nameof(value), value, "KFactor must be finite and in [1, 1000].");
            Native.faiss_IndexRefineFlat_set_k_factor(NativeHandle, value);
        }
    }
    
    private static FaissIndexHandle CreateHandle(T baseIndex, bool ownBaseIndex)
    {
        if (baseIndex.Metric != MetricType.L2 && baseIndex.Metric != MetricType.InnerProduct)
            throw new ArgumentOutOfRangeException(nameof(baseIndex), "Metric must be L2 or InnerProduct");

        if (baseIndex.TotalCount != 0)
            throw new FaissArgumentException($"{nameof(baseIndex)} must be empty");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexRefineFlat_new(out IntPtr ptr, baseIndex.Handle));

        if (ownBaseIndex)
            return new FaissIndexHandle<IndexRefineFlatRelease>(ptr);

        var keepAlive = new FaissDependentIndexHandle<IndexRefineFlatRelease>(ptr);
        keepAlive.AddDependency(baseIndex.Handle);
        return keepAlive;
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexRefineFlatRelease>(handle, ownsHandle);

    static IndexRefineFlat<T> IFromNativeIndexHandle<IndexRefineFlat<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexRefineFlat<T> IFromNativeIndexHandle<IndexRefineFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexRefineFlat<T> Clone() => ClonableFloatIndexImpl<IndexRefineFlat<T>>.Clone(this);

    public override void Dispose()
    {
        if (BaseIndex.Handle is FaissBorrowedIndexHandle)
        {
            BaseIndex.Dispose();
        }

        base.Dispose();
    }
}