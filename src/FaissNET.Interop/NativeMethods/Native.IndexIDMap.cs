using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap_own_fields(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIDMap_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap_new(out IntPtr pIndex, FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIDMap_cast(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIDMap_id_map(FaissIndexHandle index, out IntPtr pIdMap, out UIntPtr pSize);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIDMap_sub_index(FaissIndexHandle index);
}