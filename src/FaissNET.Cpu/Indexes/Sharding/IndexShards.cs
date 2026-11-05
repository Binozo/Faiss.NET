using Faiss.Cpu.Indexes.Factory;
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

public class IndexShards : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IFromNativeIndexHandle<IndexShards>
{
    private readonly Dictionary<IntPtr, INativeIndex> _views = [];
    private readonly Lock _viewsLock = new();
    private bool _viewsClosed;

    public IndexShards(int dimensions, bool threaded = true, bool successiveIds = true, bool ownIndices = true) : this(CreateHandle(dimensions, threaded, successiveIds, ownIndices), ownIndices)
    {
    }

    /// <param name="ownIndices">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    internal IndexShards(FaissIndexHandle handle, bool? ownIndices = null) : base(handle)
    {
        if (ownIndices is bool own)
        {
            OwnIndices = own;
        }
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

    public int Count => Native.faiss_IndexShards_count(NativeHandle);

    /// <summary>A view of shard <paramref name="index"/>.</summary>
    /// <remarks>
    /// This squad disposes it when it is disposed or the replica is removed, so
    /// disposing it yourself is optional (a disposed view is replaced on the next access).
    /// Add, reset and train through this squad is not recommended as writing through a view leaves their bookkeeping out of date.
    /// </remarks>
    public INativeIndex this[int index]
    {
        get
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            var member = Native.faiss_IndexShards_at(NativeHandle, index);

            lock (_viewsLock)
            {
                ObjectDisposedException.ThrowIf(_viewsClosed, this);

                if (_views.TryGetValue(member, out var view) && !view.Handle.IsClosed)
                    return view;

                return _views[member] = CreateView(member);
            }
        }
    }

    private protected virtual INativeIndex CreateView(IntPtr member) => new GenericFloatIndex(new FaissBorrowedIndexHandle(member, NativeHandle));
    
    /// <summary>
    /// Adds the index to the shards.
    /// </summary>
    /// <param name="index">
    /// The shard to add.
    /// When <see cref="OwnIndices"/> is set, these shards free it and <paramref name="index"/> is closed.
    /// Otherwise the caller keeps ownership and these shards keep it alive until they are freed or the shard is removed.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="index"/> is part of another index, such as <c>IndexIDMap.SubIndex</c>.</exception>
    /// <exception cref="InvalidOperationException">
    /// This instance neither owns its shards nor can keep them alive: it is a view of shards built with
    /// <c>ownIndices: false</c>. Add through the original instance instead.
    /// </exception>
    public virtual void AddIndex(INativeIndex index)
    {
        OwnershipGuard.RequireIndependent(index, nameof(index));

        if (Dimensions != 0 && index.Dimensions != Dimensions)
            throw new ArgumentException($"Index dimensions ({index.Dimensions}) must match the shards' dimensions ({Dimensions}).", nameof(index));

        if (Count > 0 && index is ITrainableFloatIndex trainable && trainable.IsTrained != IsTrained)
            throw trainable.IsTrained
                ? new ArgumentException("You can't add a trained index to untrained shards.", nameof(index))
                : new FaissUntrainedException();

        var owned = OwnIndices;
        var keepAlive = NativeHandle as FaissDependentIndexHandle<IndexShardsRelease>;

        if (!owned && keepAlive is null)
            throw new InvalidOperationException("Nothing would keep the shard alive: this is a view of shards that do not own them. Add through the original instance.");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_add_shard(NativeHandle, index.Handle));

        if (owned)
            index.Handle.SetHandleAsInvalid();
        else
            keepAlive!.AddDependency(index.Handle);
    }
    
    /// <summary>
    /// Removes index from the shards. The caller owns it again and may dispose it.
    /// If <see cref="OwnIndices"/> has been set to true, this operation will throw <see cref="InvalidOperationException"/>.
    /// </summary>
    /// <param name="index"></param>
    public void RemoveShard(INativeIndex index)
    {
        if (OwnIndices)
            throw new InvalidOperationException("This index can't be removed while its ownership is taken by its parent index.");

        var member = index.Handle.DangerousGetHandle();
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_remove_shard(NativeHandle, index.Handle));

        // Native no longer references it, so stop keeping it alive.
        (NativeHandle as FaissDependentIndexHandle<IndexShardsRelease>)?.RemoveDependency(index.Handle);

        lock (_viewsLock)
        {
            if (_views.Remove(member, out var view))
                view.Dispose();
        }
    }

    /// <summary>Whether this squad holds at least one shard and every shard is trained.</summary>
    public bool IsTrained
    {
        get
        {
            // We need to read from the replicas themselves as faiss clears its own flag on <see cref="FloatIndex.Reset"/>
            // although the replicas stay trained (ThreadedIndex::reset) and only restores it on the next add or train.

            var added = false;
            NativeHandle.DangerousAddRef(ref added);
            try
            {
                var count = Native.faiss_IndexShards_count(NativeHandle);
                if (count == 0)
                    return false;

                for (var i = 0; i < count; i++)
                {
                    if (Native.faiss_Index_is_trained_ptr(Native.faiss_IndexShards_at(NativeHandle, i)) == 0)
                        return false;
                }

                return true;
            }
            finally
            {
                if (added)
                    NativeHandle.DangerousRelease();
            }
        }
    }

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public virtual void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (SuccessiveIDs)
            throw new InvalidOperationException($"Can't add custom IDs to an index with {nameof(SuccessiveIDs)} enabled.");
        
        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public override void Dispose()
    {
        lock (_viewsLock)
        {
            _viewsClosed = true;

            foreach (var view in _views.Values)
                view.Dispose();

            _views.Clear();
        }

        base.Dispose();
    }

    private static FaissIndexHandle CreateHandle(int dimensions, bool threaded, bool successiveIDs, bool ownIndices)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexShards_new_with_options(out IntPtr ptr, dimensions, threaded, successiveIDs));

        // Without ownership, shards added later are kept alive by this handle until it has been freed.
        return ownIndices
            ? new FaissIndexHandle<IndexShardsRelease>(ptr)
            : new FaissDependentIndexHandle<IndexShardsRelease>(ptr);
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexShardsRelease>(handle, ownsHandle);

    static IndexShards IFromNativeIndexHandle<IndexShards>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexShards IFromNativeIndexHandle<IndexShards>.FromHandle(FaissIndexHandle handle) => new(handle);
}