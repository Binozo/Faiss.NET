using System.Runtime.InteropServices;
using Faiss.Cpu.Search;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_new(out IntPtr pIndex, long d, int nbits);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_new_with_options(out IntPtr pIndex, long d, int nbits, [MarshalAs(UnmanagedType.Bool)] bool rotateData, [MarshalAs(UnmanagedType.Bool)] bool trainThresholds);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_nbits(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_code_size(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_rotate_data(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_train_thresholds(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexLSH_free(IntPtr p_index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinary_free(IntPtr index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_d(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_is_trained(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial long faiss_IndexBinary_ntotal(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial MetricType faiss_IndexBinary_metric_type(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_verbose(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinary_set_verbose(FaissBinaryIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool verbose);

    [LibraryImport(LibraryName)] // TODO: Convert the others too!
    internal static partial int faiss_IndexBinary_train(FaissBinaryIndexHandle index, long n, byte* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_add(FaissBinaryIndexHandle index, long n, byte* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_add_with_ids(FaissBinaryIndexHandle index, long n, byte* x, long* xids);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_search(FaissBinaryIndexHandle index, long n, byte* x, long k, int* distances, long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_search_with_params(FaissBinaryIndexHandle index, long n, byte* x, long k, FaissSearchParametersHandle @params, int* distances, long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_range_search(FaissBinaryIndexHandle index, long n, byte* x, int radius, FaissRangeSearchResultHandle result);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_assign(FaissBinaryIndexHandle index, long n, byte* x, long* labels, long k);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reset(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_remove_ids(FaissBinaryIndexHandle index, FaissIDSelectorHandle sel, out nuint nRemoved);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reconstruct(FaissBinaryIndexHandle index, long key, byte* recons);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reconstruct_n(FaissBinaryIndexHandle index, long i0, long ni, byte* recons);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_clone_index_binary(FaissBinaryIndexHandle index, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_nlist(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_nprobe(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_nprobe(FaissBinaryIndexHandle index, nuint nprobe);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexBinaryIVF_quantizer(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_own_fields(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_own_fields(FaissBinaryIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_max_codes(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_max_codes(FaissBinaryIndexHandle index, nuint maxCodes);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_use_heap(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_use_heap(FaissBinaryIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool useHeap);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_per_invlist_search(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_per_invlist_search(FaissBinaryIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool perInvlistSearch);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_merge_from(FaissBinaryIndexHandle index, IntPtr other, long addId);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_get_list_size(FaissBinaryIndexHandle index, nuint listNo);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_make_direct_map(FaissBinaryIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool maintainDirectMap);

    [LibraryImport(LibraryName)]
    internal static partial double faiss_IndexBinaryIVF_imbalance_factor(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_print_stats(FaissBinaryIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_free(IntPtr p_index);
}