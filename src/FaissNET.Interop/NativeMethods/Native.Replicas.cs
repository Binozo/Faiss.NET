using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexReplicas_new(out IntPtr pIndex, long d);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexReplicas_new_with_options(out IntPtr pIndex, long d, [MarshalAs(UnmanagedType.Bool)] bool threaded);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexReplicas_add_replica(FaissIndexHandle index, FaissIndexHandle replica);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexReplicas_remove_replica(FaissIndexHandle index, FaissIndexHandle replica);
    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexReplicas_at(FaissIndexHandle index, int i);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexReplicas_own_fields(FaissIndexHandle index);
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexReplicas_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);
}