namespace Faiss.Gpu;

using System;
using System.Collections.Generic;

using Cpu;
using Interop;
using Exceptions;
using Interfaces;
using Faiss.Interfaces;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.ErrorHandling;
using Faiss.Interop.SafeHandles;

/// <summary>
/// Virtual index that wraps across multiple GPUs.
/// </summary>
public sealed class GpuFaissIndexShards<T> : FaissCpuIndex where T : INativeFaissCpuIndex
{
    private readonly FaissIndexHandle _handle;
    private protected override FaissIndexHandle NativeHandle => _handle;
    
    // Keep reference to prevent GC fuckup
    private readonly List<IFaissIndex> _shards = new();

    public GpuFaissIndexShards(int dimensions, bool threaded = true)
    {
        int isThreaded = threaded ? 1 : 0;
        
        FaissErrorHandler.ThrowIfError(GpuShardsNativeMethods.faiss_IndexShards_new_with_options(
            out IntPtr ptr, 
            dimensions, 
            isThreaded, 
            successive_ids: 1));
            
        _handle = new FaissIndexHandle(ptr);
    }

    /// <summary>
    /// Locks a GPU index into the multi-GPU pool.
    /// </summary>
    public void AddShard(INativeFaissGpuIndex<T> gpuIndex)
    {
        if (gpuIndex.Dimensions != Dimensions)
        {
            throw new ArgumentException("Dimensions mismatch. Expected: " + Dimensions + ", Actual: " + gpuIndex.Dimensions);
        }
        
        FaissErrorHandler.ThrowIfError(GpuShardsNativeMethods.faiss_IndexShards_add_shard(
            _handle.DangerousGetHandle(), 
            gpuIndex.Handle));
            
        _shards.Add(gpuIndex);
    }

    /// <summary>
    /// Forces all graphics cards to sync up.
    /// </summary>
    /// <exception cref="FaissException">Thrown when the sync operation fails.</exception>
    public void Sync()
    {
        FaissErrorHandler.ThrowIfError(GpuShardsNativeMethods.faiss_IndexShards_sync_with_shard_indexes(_handle.DangerousGetHandle()));
    }

    public new void Dispose()
    {
        _shards.Clear(); 
        base.Dispose();
    }
}