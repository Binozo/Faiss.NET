using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;
using Faiss.Cpu.Search;
using Faiss.Cpu.Selectors;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_get_last_error();

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_get_version();

    public static string FaissVersion => Marshal.PtrToStringAnsi(faiss_get_version())!;

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_d(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_is_trained(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial MetricType faiss_Index_metric_type(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_train(FaissIndexHandle index, long n, float* x);

    [LibraryImport(LibraryName)]
    internal static partial long faiss_Index_ntotal(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_add(FaissIndexHandle index, long n, float* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_search(FaissIndexHandle index, long n, float* x, long k, float* distances, long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_reset(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_verbose(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_Index_set_verbose(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool verbose);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_Index_free(IntPtr obj);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_clone_index(FaissIndexHandle index, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_reconstruct(FaissIndexHandle index, long key, float* recons);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_remove_ids(FaissIndexHandle index, FaissIDSelectorHandle sel, out nuint n_removed);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_add_with_ids(FaissIndexHandle index, long n, float* x, long* xids);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_reconstruct_n(FaissIndexHandle index, long i0, long ni, float* recons);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_assign(FaissIndexHandle index, long n, float* x, long* labels, long k);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_search_with_params(FaissIndexHandle index, long n, float* x, long k, FaissSearchParametersHandle params_ptr, float* distances, long* labels);
}