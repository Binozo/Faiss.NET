using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Sharding;

internal readonly struct IndexShardsRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexShards_free(handle);
}

public class IndexShards : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IClonableFloatIndex<IndexShards>, IFromNativeIndexHandle<IndexShards>
{
    private readonly List<INativeIndex> _shards = new();

    public IndexShards(int dimensions, bool threaded = true, bool successiveIds = true, bool ownIndices = true) : this(CreateHandle(dimensions, threaded, successiveIds), ownIndices)
    {
    }

    private IndexShards(FaissIndexHandle handle, bool ownIndices = true) : base(handle)
    {
        OwnIndices = ownIndices;
    }

    public bool OwnIndices
    {
        get => Native.faiss_IndexShards_own_indices(NativeHandle) != 0;
        private set => Native.faiss_IndexShards_set_own_indices(NativeHandle, value);
    }

    public bool SuccessiveIDs
    {
        get => Native.faiss_IndexShards_successive_ids(NativeHandle) != 0;
        private set => Native.faiss_IndexShards_set_successive_ids(NativeHandle, value);
    }
    
    /// <summary>
    /// Adds the index to the shards.
    /// </summary>
    /// <param name="index">
    /// The shard to add. When <see cref="OwnIndices"/> is set, native takes ownership and the managed wrapper
    /// is released so the shard is not freed twice.
    /// </param>
    public void AddIndex(INativeIndex index)
    {
        ArgumentNullException.ThrowIfNull(index);

        if (Dimensions != 0 && index.Dimensions != Dimensions)
            throw new ArgumentException($"Index dimensions ({index.Dimensions}) must match squad dimensions ({Dimensions})");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_add_shard(NativeHandle, index.Handle));

        if (OwnIndices)
            index.Handle.SetHandleAsInvalid();

        _shards.Add(index);
    }
    
    /// <summary>
    /// Removes index from the shards. Index must get manually disposed.
    /// </summary>
    /// <param name="index"></param>
    public void RemoveShard(INativeIndex index)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_remove_shard(NativeHandle, index.Handle));

        _shards.Remove(index);
    }
    
    private INativeIndex this[int index] => _shards[index];

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (SuccessiveIDs)
            throw new FaissException($"Can't add custom IDs to an index with {nameof(SuccessiveIDs)} enabled");
        
        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    private static FaissIndexHandle CreateHandle(int dimensions, bool threaded, bool successiveIDs)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_new_with_options(out IntPtr ptr, dimensions, threaded, successiveIDs));
        return new FaissIndexHandle<IndexShardsRelease>(ptr);
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexShardsRelease>(handle, ownsHandle);

    static IndexShards IFromNativeIndexHandle<IndexShards>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexShards IFromNativeIndexHandle<IndexShards>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    public IndexShards Clone() => ClonableFloatIndexImpl<IndexShards>.Clone(this);

    public override void Dispose()
    {
        _shards.Clear();

        base.Dispose();
        GC.SuppressFinalize(this);
    }
}