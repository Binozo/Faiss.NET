using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Sharding;

public class IndexShards : CpuIndex<IndexShards>, IFromNativeHandle<IndexShards>
{
    private readonly List<INativeIndex> _shards = new();

    public IndexShards(int dimensions, bool threaded = true, bool successiveIds = true)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexShards_new_with_options(
                out IntPtr ptr, dimensions, threaded, successiveIds)
        );

        SafeHandle = new FaissIndexHandle(ptr);
        Native.faiss_IndexShards_set_own_fields(SafeHandle, false);
    }

    private IndexShards(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexShards_set_own_fields(SafeHandle, true);
    }

    static IndexShards IFromNativeHandle<IndexShards>.FromHandle(IntPtr handle) => new(handle);
    
    /// <summary>
    /// Adds the index to the shards.
    /// </summary>
    /// <param name="index"></param>
    public void AddIndex(INativeIndex index)
    {
        if (Dimensions != 0 && index.Dimensions != Dimensions)
        {
            throw new ArgumentException($"Index dimensions ({index.Dimensions}) must match squad dimensions ({Dimensions})");
        }

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexShards_add_shard(SafeHandle, index.Handle)
        );

        _shards.Add(index);
    }
    
    /// <summary>
    /// Removes index from the shards. Index must get manually disposed.
    /// </summary>
    /// <param name="index"></param>
    public void RemoveShard(INativeIndex index)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexShards_remove_shard(SafeHandle, index.Handle)
        );

        _shards.Remove(index);
    }

    public override void Dispose()
    {
        _shards.Clear();

        base.Dispose();
        GC.SuppressFinalize(this);
    }
}