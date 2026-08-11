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

    public IndexReplicas(long dimensions, bool threaded = true, bool ownIndices = true) : this(CreateHandle(dimensions, threaded), ownIndices)
    {
    }

    private IndexReplicas(FaissIndexHandle handle, bool ownIndices = true) : base(handle)
    {
        OwnIndices = ownIndices;
    }
    
    /// <summary>
    /// Adds a cloned index to the squad.
    /// </summary>
    public void AddReplica(T replica)
    {
        if (Dimensions != 0 && replica.Dimensions != Dimensions)
        {
            throw new ArgumentException($"Replica dimensions ({replica.Dimensions}) must match squad dimensions ({Dimensions})");
        }

        if (typeof(ITrainableFloatIndex).IsAssignableFrom(typeof(T)) && _replicas.Count > 0) // TODO: Create PR that adds a count property in faiss_c
        {
            ITrainableFloatIndex first = _replicas[0] as ITrainableFloatIndex;
            ITrainableFloatIndex trainableReplica = replica as ITrainableFloatIndex;

            if (first.IsTrained != trainableReplica.IsTrained)
            {
                if (!trainableReplica.IsTrained)
                {
                    throw new FaissUntrainedException();
                }
                else
                {
                    throw new ArgumentException("You can't pass a trained index to this untrained squad of indexes.", nameof(replica));
                }
            }
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_add_replica(NativeHandle, replica.Handle));

        _replicas.Add(replica);
    }

    /// <summary>
    /// Removes a cloned index from the squad.
    /// </summary>
    public void RemoveReplica(T replica)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_remove_replica(NativeHandle, replica.Handle));

        _replicas.Remove(replica);
    }

    public bool OwnIndices
    {
        get => Native.faiss_IndexReplicas_own_indices(NativeHandle) != 0;
        private set => Native.faiss_IndexReplicas_set_own_indices(NativeHandle, value);
    }

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialFloatIndex)this).Add(count, vectors);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    private static FaissIndexHandle CreateHandle(long dimensions, bool threaded)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_new_with_options(out IntPtr ptr, dimensions, threaded));
        return new FaissIndexHandle<IndexReplicasRelease>(ptr);
    }

    static IndexReplicas<T> IFromNativeIndexHandle<IndexReplicas<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public override void Dispose()
    {
        _replicas.Clear();

        base.Dispose();
        GC.SuppressFinalize(this);
    }
}