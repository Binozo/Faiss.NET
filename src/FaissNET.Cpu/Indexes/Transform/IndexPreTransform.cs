using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Cpu.Transforms;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Transform;

internal readonly struct IndexPreTransformRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexPreTransform_free(handle);
}

/// <summary>
/// Wraps an index and applies a chain of vector transforms before adding or searching vectors.
/// </summary>
public sealed class IndexPreTransform<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IRangeSearchFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IFromNativeIndexHandle<IndexPreTransform<T>>, IClonableFloatIndex<IndexPreTransform<T>> where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    public readonly T Index;
    private readonly List<VectorTransform> _chain = new();

    public IndexPreTransform(T index, VectorTransform transform, bool ownSubIndex = false) : this(CreateHandle(index, transform), ownSubIndex)
    {
        if (ownSubIndex)
        {
            index.Handle.SetHandleAsInvalid();
            transform.ReleaseOwnership();
        }

        _chain.Add(transform);
    }

    public IndexPreTransform(T index, bool ownSubIndex = false) : this(CreateHandle(index), ownSubIndex)
    {
        if (ownSubIndex)
        {
            index.Handle.SetHandleAsInvalid();
        }
    }

    private IndexPreTransform(FaissIndexHandle handle, bool ownSubIndex = true) : base(handle)
    {
        OwnSubIndex = ownSubIndex;
        Index = T.FromPointer(Native.faiss_IndexPreTransform_index(handle), false);
    }

    public bool OwnSubIndex
    {
        get => Native.faiss_IndexPreTransform_own_fields(NativeHandle) != 0;
        private set =>  Native.faiss_IndexPreTransform_set_own_fields(NativeHandle, value);
    }

    /// <summary>
    /// Prepends a transform to the chain. The prepended transform is applied first during add and search operations.
    /// </summary>
    /// <param name="transform">The transform to prepend.</param>
    public void PrependTransform(VectorTransform transform)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexPreTransform_prepend_transform(NativeHandle, transform.Handle));
        _chain.Insert(0, transform);

        if (OwnSubIndex)
        {
            transform.ReleaseOwnership();
        }
    }

    /// <summary>
    /// Gets the transform chain in application order.
    /// </summary>
    public IReadOnlyList<VectorTransform> TransformChain => _chain.AsReadOnly();

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    private static FaissIndexHandle CreateHandle(T subIndex, VectorTransform? transform = null)
    {
        IntPtr ptr;
        FaissErrorHandler.ThrowIfError(transform != null
            ? Native.faiss_IndexPreTransform_new_with_transform(out ptr, transform.Handle, subIndex.Handle)
            : Native.faiss_IndexPreTransform_new_with(out ptr, subIndex.Handle));

        return new FaissIndexHandle<IndexPreTransformRelease>(ptr);
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexPreTransformRelease>(handle, ownsHandle);

    static IndexPreTransform<T> IFromNativeIndexHandle<IndexPreTransform<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));
    
    static IndexPreTransform<T> IFromNativeIndexHandle<IndexPreTransform<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexPreTransform<T> Clone() => ClonableFloatIndexImpl<IndexPreTransform<T>>.Clone(this);
}