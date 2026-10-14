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
public sealed class IndexPreTransform<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IRangeSearchFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IFromNativeIndexHandle<IndexPreTransform<T>>, IClonableFloatIndex<IndexPreTransform<T>>, IKeyLayout where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    /// <summary>The wrapped index.</summary>
    /// <remarks>
    /// With <c>ownSubIndex: true</c>, this is a view of the sub-index now owned by the
    /// <see cref="IndexPreTransform{T}"/>; it is valid until the <see cref="IndexPreTransform{T}"/> is disposed.
    /// With <c>ownSubIndex: false</c>, the caller keeps ownership, and the <see cref="IndexPreTransform{T}"/>
    /// keeps the native sub-index alive until it is itself disposed, even if the caller disposes its reference first.
    /// </remarks>
    public readonly T Index;
    private readonly List<VectorTransform> _chain = new();

    public IndexPreTransform(T index, VectorTransform transform, bool ownSubIndex = false) : this(CreateHandle(OwnershipGuard.RequireIndependent(index, nameof(index)), RequireOwned(transform), ownSubIndex), ownSubIndex, ownSubIndex ? null : index)
    {
        if (ownSubIndex)
        {
            // This parent index is responsible for freeing both now
            // Store only the view in the chain
            _chain.Add(transform.CreateView(NativeHandle));
            index.Handle.SetHandleAsInvalid();
            transform.Handle.SetHandleAsInvalid();
        }
        else
        {
            _chain.Add(transform);
        }
    }

    private static VectorTransform RequireOwned(VectorTransform transform)
    {
        ArgumentNullException.ThrowIfNull(transform);

        if (transform.Handle is FaissBorrowedVectorTransformHandle)
            throw new ArgumentException("This transform belongs to another index (it came from a TransformChain); create a new one instead.", nameof(transform));

        return transform;
    }

    public IndexPreTransform(T index, bool ownSubIndex = false) : this(CreateHandle(OwnershipGuard.RequireIndependent(index, nameof(index)), transform: null, ownSubIndex), ownSubIndex, ownSubIndex ? null : index)
    {
        if (ownSubIndex)
            index.Handle.SetHandleAsInvalid();
    }

    /// <param name="ownSubIndex">The ownership the caller requested. <see langword="null"/> when wrapping, so the native flag is left as faiss set it.</param>
    /// <param name="callerIndex">Without ownership: the caller's sub-index, exposed as-is. Otherwise, a view is created.</param>
    private IndexPreTransform(FaissIndexHandle handle, bool? ownSubIndex = null, T? callerIndex = null) : base(handle)
    {
        if (ownSubIndex is bool own)
            OwnSubIndex = own;

        Index = callerIndex ?? T.FromBorrowed(Native.faiss_IndexPreTransform_index(handle), handle);
    }

    public bool OwnSubIndex
    {
        get => Native.faiss_IndexPreTransform_own_fields(NativeHandle) != 0;
        private set =>  Native.faiss_IndexPreTransform_set_own_fields(NativeHandle, value);
    }

    /// <summary>
    /// Prepends a transform to the chain. The prepended transform is applied first during add and search operations.
    /// </summary>
    /// <param name="transform">
    /// The transform to prepend. When this index owns its parts (<see cref="OwnSubIndex"/>), ownership passes to it and <paramref name="transform"/> is closed.
    /// Otherwise, the caller keeps ownership and this index keeps the transform alive until this index is freed.
    /// </param>
    /// <exception cref="InvalidOperationException">
    /// This instance neither owns its parts nor can keep the transform alive: it is a non-owning view of an index
    /// built with <c>ownSubIndex: false</c>. Prepend through the original instance instead.
    /// </exception>
    /// <exception cref="ArgumentException"><paramref name="transform"/> came from another index's <see cref="TransformChain"/>.</exception>
    public void PrependTransform(VectorTransform transform)
    {
        RequireOwned(transform);

        var owned = OwnSubIndex;
        var keepAlive = NativeHandle as FaissDependentIndexHandle<IndexPreTransformRelease>;

        if (!owned && keepAlive is null)
            throw new InvalidOperationException("Nothing would keep the transform alive: this is a view of an index that does not own its parts. Prepend through the original instance.");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexPreTransform_prepend_transform(NativeHandle, transform.Handle));

        if (owned)
        {
            _chain.Insert(0, transform.CreateView(NativeHandle));
            transform.Handle.SetHandleAsInvalid();
        }
        else
        {
            keepAlive!.AddDependency(transform.Handle);
            _chain.Insert(0, transform);
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

    KeyLayout IKeyLayout.KeyLayout => KeyLayout.Of(Index);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    private static FaissIndexHandle CreateHandle(T subIndex, VectorTransform? transform, bool ownSubIndex)
    {
        IntPtr ptr;
        FaissErrorHandler.ThrowIfError(transform != null
            ? Native.faiss_IndexPreTransform_new_with_transform(out ptr, transform.Handle, subIndex.Handle)
            : Native.faiss_IndexPreTransform_new_with(out ptr, subIndex.Handle));

        if (ownSubIndex)
            return new FaissIndexHandle<IndexPreTransformRelease>(ptr);

        // As the caller owns the parts they must be kept alive until this index has been freed
        var handle = new FaissDependentIndexHandle<IndexPreTransformRelease>(ptr);
        handle.AddDependency(subIndex.Handle);

        if (transform != null)
            handle.AddDependency(transform.Handle);

        return handle;
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexPreTransformRelease>(handle, ownsHandle);

    static IndexPreTransform<T> IFromNativeIndexHandle<IndexPreTransform<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));
    
    static IndexPreTransform<T> IFromNativeIndexHandle<IndexPreTransform<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexPreTransform<T> Clone() => ClonableFloatIndexImpl<IndexPreTransform<T>>.Clone(this);

    public override void Dispose()
    {
        // As borrowed/views of the sub-index and chain transforms reference this index's own handle,
        // they need to be freed first to ensure this index gets disposed now
        if (Index.Handle is FaissBorrowedIndexHandle)
            Index.Dispose();

        foreach (var transform in _chain)
        {
            if (transform.Handle is FaissBorrowedVectorTransformHandle)
                transform.Dispose();
        }

        base.Dispose();
    }
}