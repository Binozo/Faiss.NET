namespace Faiss.Gpu.Interop;

using System;
using System.Runtime.InteropServices;

internal static partial class GpuShardsNativeMethods
{
    private const string LibraryName = "faiss_c";

    // --- Index Sharding ---

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_new(out IntPtr p_index, int d);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_new_with_options(out IntPtr p_index, int d, int threaded, int successive_ids);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_add_shard(IntPtr index, IntPtr shard);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_sync_with_shard_indexes(IntPtr index);
}