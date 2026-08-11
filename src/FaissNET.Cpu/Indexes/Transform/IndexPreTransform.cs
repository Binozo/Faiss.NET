using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Transforms;
using Faiss.Exceptions;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Transform;

internal readonly struct IndexPreTransformRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexPreTransform_free(handle);
}

/// <summary>
/// Wraps an index and applies a chain of vector transforms before adding or searching vectors.
/// </summary>
public sealed class IndexPreTransform<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IRangeSearchFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, IFromNativeIndexHandle<IndexPreTransform<T>>, IClonableFloatIndex<IndexPreTransform<T>> where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
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

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    private static FaissIndexHandle CreateHandle(T subIndex, VectorTransform? transform = null)
    {
        IntPtr ptr;
        FaissErrorHandler.ThrowIfError(transform != null
            ? Native.faiss_IndexPreTransform_new_with_transform(out ptr, transform.Handle, subIndex.Handle)
            : Native.faiss_IndexPreTransform_new_with(out ptr, subIndex.Handle));

        return new FaissIndexHandle<IndexPreTransformRelease>(ptr);
    }
    
    static IndexPreTransform<T> IFromNativeIndexHandle<IndexPreTransform<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexPreTransform<T> Clone() => ((IClonableFloatIndex<IndexPreTransform<T>>)this).Clone();
}