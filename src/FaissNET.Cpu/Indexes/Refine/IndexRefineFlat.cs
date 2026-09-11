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
    private readonly bool _ownBaseIndex;

    public IndexRefineFlat(T baseIndex, bool ownBaseIndex = true) : this(CreateHandle(baseIndex), baseIndex, ownBaseIndex)
    {
        if (ownBaseIndex)
        {
            baseIndex.Handle.SetHandleAsInvalid();
        }
    }
    
    internal IndexRefineFlat(FaissIndexHandle handle, T? baseIndex = null, bool ownBaseIndex = true) : base(handle)
    {
        BaseIndex = baseIndex ?? T.FromPointer(Native.faiss_IndexRefineFlat_base_index(NativeHandle));
        OwnBaseIndex = ownBaseIndex;
        _ownBaseIndex = ownBaseIndex;
    }

    public bool OwnBaseIndex
    {
        get => Native.faiss_IndexRefineFlat_own_fields(BaseIndex.Handle) != 0;
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
    /// Oversampling factor for refinement. Must be >= 1.
    /// When searching for k results, the base index is queried for k * KFactor candidates,
    /// which are then re-ranked exactly by the internal flat index.
    /// </summary>
    public float KFactor
    {
        get => Native.faiss_IndexRefineFlat_k_factor(NativeHandle);
        set
        {
            if (value < 1.0f)
                throw new ArgumentOutOfRangeException(nameof(value), "KFactor must be >= 1");
            Native.faiss_IndexRefineFlat_set_k_factor(NativeHandle, value);
        }
    }
    
    private static FaissIndexHandle CreateHandle(T baseIndex)
    {
        if (baseIndex.Metric != MetricType.L2 && baseIndex.Metric != MetricType.InnerProduct)
            throw new ArgumentOutOfRangeException(nameof(baseIndex), "Metric must be L2 or InnerProduct");

        if (baseIndex.TotalCount != 0)
            throw new FaissArgumentException($"{nameof(baseIndex)} must be empty");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexRefineFlat_new(out IntPtr ptr, baseIndex.Handle));
        return new FaissIndexHandle<IndexRefineFlatRelease>(ptr);
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexRefineFlatRelease>(handle, ownsHandle);

    static IndexRefineFlat<T> IFromNativeIndexHandle<IndexRefineFlat<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexRefineFlat<T> IFromNativeIndexHandle<IndexRefineFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexRefineFlat<T> Clone() => ClonableFloatIndexImpl<IndexRefineFlat<T>>.Clone(this);

    public override void Dispose()
    {
        if (_ownBaseIndex)
        {
            BaseIndex.Dispose();
        }

        base.Dispose();
    }
}