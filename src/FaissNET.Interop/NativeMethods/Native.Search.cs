using System.Runtime.InteropServices;
using Faiss.Cpu.Search;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParameters_new(out IntPtr pSp, FaissIDSelectorHandle? sel);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParameters_free(IntPtr sp);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersIVF_new_with(out IntPtr pSp, FaissIDSelectorHandle? sel, UIntPtr nprobe, UIntPtr maxCodes);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_free(IntPtr sp);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_SearchParametersIVF_nprobe(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_set_nprobe(FaissSearchParametersHandle sp, UIntPtr nprobe);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_SearchParametersIVF_max_codes(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_set_max_codes(FaissSearchParametersHandle sp, UIntPtr maxCodes);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_SearchParametersIVF_max_lists_num(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_set_max_lists_num(FaissSearchParametersHandle sp, UIntPtr maxLists);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersIVF_ensure_topk_full(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_set_ensure_topk_full(FaissSearchParametersHandle sp, int ensureTopkFull);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_SearchParametersIVF_max_empty_result_buckets(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersIVF_set_max_empty_result_buckets(FaissSearchParametersHandle sp, UIntPtr maxEmptyResultBuckets);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersHNSW_new_with(out IntPtr pSp, IntPtr sel, int efSearch);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersHNSW_free(IntPtr sp);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersHNSW_efSearch(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersHNSW_set_efSearch(FaissSearchParametersHandle sp, int efSearch);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersHNSW_check_relative_distance(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersHNSW_set_check_relative_distance(FaissSearchParametersHandle sp, [MarshalAs(UnmanagedType.Bool)] bool value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_SearchParametersHNSW_bounded_queue(FaissSearchParametersHandle sp);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_SearchParametersHNSW_set_bounded_queue(FaissSearchParametersHandle sp, [MarshalAs(UnmanagedType.Bool)] bool value);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_range_search(FaissIndexHandle index, long n, float* x, float radius, FaissRangeSearchResultHandle result);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_RangeSearchResult_new(out IntPtr pRsr, long nq);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_RangeSearchResult_nq(FaissRangeSearchResultHandle rsr);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_RangeSearchResult_lims(FaissRangeSearchResultHandle rsr, out IntPtr lims);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_RangeSearchResult_labels(FaissRangeSearchResultHandle rsr, out IntPtr labels, out IntPtr distances);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_RangeSearchResult_free(IntPtr rsr);
}