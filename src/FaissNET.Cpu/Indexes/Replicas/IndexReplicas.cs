using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Replicas;

internal readonly struct IndexReplicasRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexReplicas_free(handle);
}

public class IndexReplicas<T> : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, IFromNativeIndexHandle<IndexReplicas<T>> where T : FloatIndex, IFromNativeIndexHandle<T>, INativeIndex
{
    private readonly List<T> _replicas = new();

    public IndexReplicas(long dimensions, bool threaded = true, bool ownIndices = true) : this(CreateHandle(dimensions, threaded, ownIndices), ownIndices)
    {
    }

    /// <param name="ownIndices">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    private IndexReplicas(FaissIndexHandle handle, bool? ownIndices = null) : base(handle)
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
    public void AddReplica(T replica)
    {
        OwnershipGuard.RequireIndependent(replica, nameof(replica));

        if (Dimensions != 0 && replica.Dimensions != Dimensions)
        {
            throw new ArgumentException($"Replica dimensions ({replica.Dimensions}) must match squad dimensions ({Dimensions})");
        }
        
        // TODO: Don't use _replicas.Count for counting (won't work for IndexReplicas got from FromPointer) and use c api instead
        // Waiting for PR #5512 (https://github.com/facebookresearch/faiss/pull/5512) to merge
        if (_replicas.Count > 0 && replica is ITrainableFloatIndex trainable && trainable.IsTrained != IsTrained)
            throw trainable.IsTrained
                ? new ArgumentException("You can't pass a trained index to this untrained squad of indexes.", nameof(replica))
                : new FaissUntrainedException();

        // Decide who keeps the replica alive before native takes the pointer, so a refusal leaves nothing half-done.
        var owned = OwnIndices;
        var keepAlive = NativeHandle as FaissDependentIndexHandle<IndexReplicasRelease>;

        if (!owned && keepAlive is null)
            throw new InvalidOperationException("Nothing would keep the replica alive: this is a view of a squad that does not own its replicas. Add through the original instance.");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_add_replica(NativeHandle, replica.Handle));

        if (owned)
            replica.Handle.SetHandleAsInvalid();
        else
            keepAlive!.AddDependency(replica.Handle);

        _replicas.Add(replica);
    }

    /// <summary>
    /// Removes a cloned index from the squad. The caller owns it again and may dispose it.
    /// If <see cref="OwnIndices"/> has been set to true, this operation will throw <see cref="InvalidOperationException"/>.
    /// </summary>
    public void RemoveReplica(T replica)
    {
        if (OwnIndices)
            throw new InvalidOperationException("This index can't be removed while its ownership is taken by its parent index.");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_remove_replica(NativeHandle, replica.Handle));

        // Native no longer references it, so stop keeping it alive.
        (NativeHandle as FaissDependentIndexHandle<IndexReplicasRelease>)?.RemoveDependency(replica.Handle);

        _replicas.Remove(replica);
    }

    public bool OwnIndices
    {
        get => Native.faiss_IndexReplicas_own_indices(NativeHandle) != 0;
        private set => Native.faiss_IndexReplicas_set_own_indices(NativeHandle, value);
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

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

    public override void Dispose()
    {
        _replicas.Clear();

        base.Dispose();
        GC.SuppressFinalize(this);
    }
}