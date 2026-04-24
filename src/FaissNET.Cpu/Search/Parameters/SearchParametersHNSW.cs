using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Search.Parameters;

public sealed class SearchParametersHNSW : SearchParameters
{
    public SearchParametersHNSW(int efSearch, IDSelector? selector = null)
        : base(CreateHandle(efSearch, selector))
    {
    }
    
    private static IntPtr CreateHandle(int efSearch, IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParametersHNSW_new_with(out IntPtr ptr, selector?.SafeHandle.DangerousGetHandle() ?? IntPtr.Zero, efSearch)
        );

        return ptr;
    }
    
    public int EfSearch
    {
        get => Native.faiss_SearchParametersHNSW_efSearch(SafeHandle);
        set => Native.faiss_SearchParametersHNSW_set_efSearch(SafeHandle, value);
    }
    
    public bool CheckRelativeDistance
    {
        get => Native.faiss_SearchParametersHNSW_check_relative_distance(SafeHandle) != 0;
        set => Native.faiss_SearchParametersHNSW_set_check_relative_distance(SafeHandle, value);
    }
    
    public bool BoundedQueue
    {
        get => Native.faiss_SearchParametersHNSW_bounded_queue(SafeHandle) != 0;
        set => Native.faiss_SearchParametersHNSW_set_bounded_queue(SafeHandle, value);
    }
}