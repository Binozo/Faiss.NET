using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Search.Parameters;

public sealed class SearchParametersIVF : SearchParameters
{
    public SearchParametersIVF(int nprobe, int maxCodes = 0, IDSelector? selector = null) 
        : base(CreateHandle(nprobe, maxCodes, selector))
    {
    }
    
    private static IntPtr CreateHandle(int nprobe, int maxCodes, IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParametersIVF_new_with(out IntPtr ptr, selector?.SafeHandle.DangerousGetHandle() ?? IntPtr.Zero, (UIntPtr)nprobe, (UIntPtr)maxCodes)
        );
    
        return ptr;
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