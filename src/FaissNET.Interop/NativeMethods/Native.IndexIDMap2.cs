using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap2_own_fields(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIDMap2_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap2_new(out IntPtr pIndex, FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap2_construct_rev_map(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIDMap2_cast(IntPtr index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIDMap2_id_map(FaissIndexHandle index, out IntPtr pIdMap, out UIntPtr pSize);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIDMap2_sub_index(FaissIndexHandle index);
}