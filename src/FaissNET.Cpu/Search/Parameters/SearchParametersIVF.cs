using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Search.Parameters;

internal readonly struct SearchParametersIVFRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_SearchParametersIVF_free(handle);
}

public sealed class SearchParametersIVF : SearchParameters
{
    /// <summary>Per-query IVF search parameters.</summary>
    /// <param name="nprobe">Lists to probe. Default-constructed native IVF params use 1, not the index's nprobe.</param>
    /// <param name="maxCodes">Max codes to scan; 0 is unlimited.</param>
    /// <param name="selector">
    /// Ids to consider during search, or <see langword="null"/> for no filter.
    /// Must remain undisposed for the lifetime of this instance.
    /// </param>
    /// <remarks>
    /// This object borrows <paramref name="selector"/>. Do not dispose the selector while these parameters are in use.
    /// A base <see cref="SearchParameters"/> is not valid for IVF search; use this type.
    /// </remarks>
    public SearchParametersIVF(int nprobe, int maxCodes = 0, IDSelector? selector = null) 
        : base(CreateHandle(nprobe, maxCodes, selector), selector)
    {
    }
    
    private static FaissSearchParametersHandle CreateHandle(int nprobe, int maxCodes, IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParametersIVF_new_with(out IntPtr ptr, selector?.SafeHandle, (UIntPtr)nprobe, (UIntPtr)maxCodes)
        );
    
        return new FaissSearchParametersHandle<SearchParametersIVFRelease>(ptr);
    }
    
    public int Nprobe
    {
        get => (int)Native.faiss_SearchParametersIVF_nprobe(SafeHandle);
        set => Native.faiss_SearchParametersIVF_set_nprobe(SafeHandle, (UIntPtr)value);
    }
    
    public int MaxCodes
    {
        get => (int)Native.faiss_SearchParametersIVF_max_codes(SafeHandle);
        set => Native.faiss_SearchParametersIVF_set_max_codes(SafeHandle, (UIntPtr)value);
    }
    
    public int MaxListsNum
    {
        get => (int)Native.faiss_SearchParametersIVF_max_lists_num(SafeHandle);
        set => Native.faiss_SearchParametersIVF_set_max_lists_num(SafeHandle, (UIntPtr)value);
    }
    
    public int EnsureTopKFull
    {
        get => Native.faiss_SearchParametersIVF_ensure_topk_full(SafeHandle);
        set => Native.faiss_SearchParametersIVF_set_ensure_topk_full(SafeHandle, value);
    }
    
    public int MaxEmptyResultBuckets
    {
        get => (int)Native.faiss_SearchParametersIVF_max_empty_result_buckets(SafeHandle);
        set => Native.faiss_SearchParametersIVF_set_max_empty_result_buckets(SafeHandle, (UIntPtr)value);
    }
}