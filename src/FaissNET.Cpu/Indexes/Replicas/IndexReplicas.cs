using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Replicas;

internal readonly struct IndexReplicasRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexReplicas_free(handle);
}

public class IndexReplicas<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, IFromNativeIndexHandle<IndexReplicas<T>>, IKeyLayout where T : FloatIndex, IFromNativeIndexHandle<T>, INativeIndex
{
    private readonly Dictionary<IntPtr, T> _views = [];
    private readonly Lock _viewsLock = new();
    private bool _viewsClosed;

    KeyLayout IKeyLayout.KeyLayout => Count > 0 && KeyLayout.Of(this[0]).KeysAreIds ? KeyLayout.Ids : KeyLayout.Positions;

    public IndexReplicas(long dimensions, bool threaded = true, bool ownIndices = true) : this(CreateHandle(dimensions, threaded, ownIndices), ownIndices)
    {
    }

    /// <param name="ownIndices">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    protected internal IndexReplicas(FaissIndexHandle handle, bool? ownIndices = null) : base(handle)
    {
        if (ownIndices is bool own)
        {
            OwnIndices = own;
        }
    }
    
    /// <summary>
    /// Adds a cloned index to the squad.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="replica"/> is part of another index, such as <c>IndexIDMap.SubIndex</c>.</exception>
    /// <exception cref="InvalidOperationException">
    /// This instance neither owns its replicas nor can keep them alive: it is a view of a squad built with
    /// <c>ownIndices: false</c>. Add through the original instance instead.
    /// </exception>
    public virtual void AddReplica(T replica)
    {
        OwnershipGuard.RequireIndependent(replica, nameof(replica));

        if (Dimensions != 0 && replica.Dimensions != Dimensions)
        {
            throw new ArgumentException($"Replica dimensions ({replica.Dimensions}) must match squad dimensions ({Dimensions})");
        }

        if (Count > 0)
        {
            if (replica is ITrainableFloatIndex trainable && trainable.IsTrained != IsTrained)
                throw trainable.IsTrained
                    ? new ArgumentException("You can't pass a trained index to this untrained squad of indexes.", nameof(replica))
                    : new FaissUntrainedException();

            if (replica.TotalCount != TotalCount)
                throw new ArgumentException($"Replica vector count ({replica.TotalCount}) must match squad vector count ({TotalCount})", nameof(replica));
        }

        var owned = OwnIndices;
        var keepAlive = NativeHandle as FaissDependentIndexHandle<IndexReplicasRelease>;

        if (!owned && keepAlive is null)
            throw new InvalidOperationException("Nothing would keep the replica alive: this is a view of a squad that does not own its replicas. Add through the original instance.");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_add_replica(NativeHandle, replica.Handle));

        if (owned)
            replica.Handle.SetHandleAsInvalid();
        else
            keepAlive!.AddDependency(replica.Handle);
    }

    /// <summary>
    /// Removes a cloned index from the squad. The caller owns it again and may dispose it.
    /// If <see cref="OwnIndices"/> has been set to true, this operation will throw <see cref="InvalidOperationException"/>.
    /// </summary>
    public void RemoveReplica(T replica)
    {
        if (OwnIndices)
            throw new InvalidOperationException("This index can't be removed while its ownership is taken by its parent index.");

        var member = replica.Handle.DangerousGetHandle();
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_remove_replica(NativeHandle, replica.Handle));

        // Native no longer references it, so stop keeping it alive.
        (NativeHandle as FaissDependentIndexHandle<IndexReplicasRelease>)?.RemoveDependency(replica.Handle);

        lock (_viewsLock)
        {
            if (_views.Remove(member, out var view))
                view.Dispose();
        }
    }

    public int Count => Native.faiss_IndexReplicas_count(NativeHandle);

    /// <summary>A view of replica <paramref name="index"/>.</summary>
    /// <remarks>
    /// This squad disposes it when it is disposed or the replica is removed, so
    /// disposing it yourself is optional (a disposed view is replaced on the next access).
    /// Add, reset and train through this squad is not recommended as writing through a view makes the replicas diverge.
    /// </remarks>
    public T this[int index]
    {
        get
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            var member = Native.faiss_IndexReplicas_at(NativeHandle, index);

            lock (_viewsLock)
            {
                ObjectDisposedException.ThrowIf(_viewsClosed, this);

                if (_views.TryGetValue(member, out var view) && !view.Handle.IsClosed)
                    return view;

                return _views[member] = T.FromBorrowed(member, NativeHandle);
            }
        }
    }

    public bool OwnIndices
    {
        get => Native.faiss_IndexReplicas_own_indices(NativeHandle) != 0;
        private set => Native.faiss_IndexReplicas_set_own_indices(NativeHandle, value);
    }

    /// <summary>Whether this squad holds at least one replica and every replica is trained.</summary>
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
                var count = Native.faiss_IndexReplicas_count(NativeHandle);
                if (count == 0)
                    return false;

                for (var i = 0; i < count; i++)
                {
                    if (Native.faiss_Index_is_trained_ptr(Native.faiss_IndexReplicas_at(NativeHandle, i)) == 0)
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

    public virtual void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => IDMappedFloatIndexImpl.Add(this, count, vectors, xids);

    public virtual long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

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

    private static FaissIndexHandle CreateHandle(long dimensions, bool threaded, bool ownIndices)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_new_with_options(out IntPtr ptr, dimensions, threaded));

        return ownIndices
            ? new FaissIndexHandle<IndexReplicasRelease>(ptr)
            : new FaissDependentIndexHandle<IndexReplicasRelease>(ptr);
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexReplicasRelease>(handle, ownsHandle);

    static IndexReplicas<T> IFromNativeIndexHandle<IndexReplicas<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexReplicas<T> IFromNativeIndexHandle<IndexReplicas<T>>.FromHandle(FaissIndexHandle handle) => new(handle);
}