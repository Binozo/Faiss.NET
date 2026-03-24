namespace Faiss.Interop.NativeMethods;

using System.Runtime.InteropServices;

using Models;
using SafeHandles;

internal static unsafe partial class Native
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_get_last_error();

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatL2_new(out FaissIndexHandle p_index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatL2_new_with(out FaissIndexHandle p_index, long d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatIP_new(out FaissIndexHandle p_index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatIP_new_with(out FaissIndexHandle p_index, long d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_d(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_is_trained(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial MetricType faiss_Index_metric_type(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial long faiss_Index_ntotal(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_add(FaissIndexHandle index, long n, float* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_search(
        FaissIndexHandle index, 
        long n, 
        float* x, 
        long k, 
        float* distances, 
        long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_reset(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_Index_free(IntPtr obj);
}