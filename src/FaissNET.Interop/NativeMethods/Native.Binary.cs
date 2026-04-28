using System.Runtime.InteropServices;
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
    internal static partial void faiss_IndexBinary_free(IntPtr index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_d(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_is_trained(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial long faiss_IndexBinary_ntotal(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial MetricType faiss_IndexBinary_metric_type(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_verbose(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinary_set_verbose(FaissIndexBinaryHandle index, [MarshalAs(UnmanagedType.Bool)] bool verbose);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_train(FaissIndexBinaryHandle index, long n, byte* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_add(FaissIndexBinaryHandle index, long n, byte* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_add_with_ids(FaissIndexBinaryHandle index, long n, byte* x, long* xids);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_search(FaissIndexBinaryHandle index, long n, byte* x, long k, int* distances, long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_search_with_params(FaissIndexBinaryHandle index, long n, byte* x, long k, IntPtr @params, int* distances, long* labels);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_range_search(FaissIndexBinaryHandle index, long n, byte* x, int radius, IntPtr result);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_assign(FaissIndexBinaryHandle index, long n, byte* x, long* labels, long k);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reset(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_remove_ids(FaissIndexBinaryHandle index, IntPtr sel, out UIntPtr nRemoved);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reconstruct(FaissIndexBinaryHandle index, long key, byte* recons);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinary_reconstruct_n(FaissIndexBinaryHandle index, long i0, long ni, byte* recons);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_clone_index_binary(FaissIndexBinaryHandle index, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_nlist(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_nprobe(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_nprobe(FaissIndexBinaryHandle index, UIntPtr nprobe);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexBinaryIVF_quantizer(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_own_fields(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_own_fields(FaissIndexBinaryHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_max_codes(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_max_codes(FaissIndexBinaryHandle index, UIntPtr maxCodes);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_use_heap(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_use_heap(FaissIndexBinaryHandle index, [MarshalAs(UnmanagedType.Bool)] bool useHeap);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_per_invlist_search(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_set_per_invlist_search(FaissIndexBinaryHandle index, [MarshalAs(UnmanagedType.Bool)] bool perInvlistSearch);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_merge_from(FaissIndexBinaryHandle index, IntPtr other, long addId);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexBinaryIVF_get_list_size(FaissIndexBinaryHandle index, UIntPtr listNo);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexBinaryIVF_make_direct_map(FaissIndexBinaryHandle index, [MarshalAs(UnmanagedType.Bool)] bool maintainDirectMap);

    [LibraryImport(LibraryName)]
    internal static partial double faiss_IndexBinaryIVF_imbalance_factor(FaissIndexBinaryHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexBinaryIVF_print_stats(FaissIndexBinaryHandle index);
}