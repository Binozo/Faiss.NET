using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_new(out IntPtr pIndex, long d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_new_with_options(out IntPtr pIndex, long d, [MarshalAs(UnmanagedType.Bool)] bool threaded, [MarshalAs(UnmanagedType.Bool)] bool successiveIds);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_add_shard(FaissIndexHandle index, FaissIndexHandle shard);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_remove_shard(FaissIndexHandle index, FaissIndexHandle shard);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexShards_at(FaissIndexHandle index, int i);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_own_indices(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexShards_set_own_indices(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexShards_successive_ids(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexShards_set_successive_ids(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool successiveIds);
}