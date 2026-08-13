using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexRefineFlat_new(out IntPtr pIndex, FaissIndexHandle baseIndex);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexRefineFlat_cast(IntPtr index);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexRefineFlat_base_index(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexRefineFlat_own_fields(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexRefineFlat_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial float faiss_IndexRefineFlat_k_factor(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexRefineFlat_set_k_factor(FaissIndexHandle index, float kFactor);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexRefineFlat_free(IntPtr index);
}