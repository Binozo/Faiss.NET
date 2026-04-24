using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Replicas;

public class IndexReplicas<T> : CpuIndex<IndexReplicas<T>>, IFromNativeHandle<IndexReplicas<T>> where T : CpuIndex<T>, IFromNativeHandle<T>, INativeIndex
{
    private readonly List<T> _replicas = new();

    public IndexReplicas(long dimensions, bool threaded = true)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexReplicas_new_with_options(out var handle, dimensions, threaded));

        SafeHandle = new FaissIndexHandle(handle);
        Native.faiss_IndexReplicas_set_own_fields(SafeHandle, false);
    }

    private IndexReplicas(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexReplicas_set_own_fields(SafeHandle, true);
    }

    static IndexReplicas<T> IFromNativeHandle<IndexReplicas<T>>.FromHandle(IntPtr handle) => new(handle);
    
    /// <summary>
    /// Adds a cloned index to the squad.
    /// </summary>
    public void AddReplica(T replica)
    {
        if (Dimensions != 0 && replica.Dimensions != Dimensions)
        {
            throw new ArgumentException($"Replica dimensions ({replica.Dimensions}) must match squad dimensions ({Dimensions})");
        }

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexReplicas_add_replica(SafeHandle, replica.Handle)
        );

        _replicas.Add(replica);
    }

    /// <summary>
    /// Removes a cloned index from the squad.
    /// </summary>
    public void RemoveReplica(T replica)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexReplicas_remove_replica(SafeHandle, replica.Handle)
        );

        _replicas.Remove(replica);
    }

    public override void Dispose()
    {
        _replicas.Clear();

        base.Dispose();
        GC.SuppressFinalize(this);
    }
}