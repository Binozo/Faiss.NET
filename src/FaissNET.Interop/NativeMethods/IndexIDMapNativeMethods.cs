namespace Faiss.Interop.NativeMethods;

using System.Runtime.InteropServices;

internal static partial class IndexIDMapNativeMethods
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIDMap_new(out IntPtr p_out, IntPtr index);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_add_with_ids(
        IntPtr index, 
        long n, 
        float* x, 
        long* xids);
}