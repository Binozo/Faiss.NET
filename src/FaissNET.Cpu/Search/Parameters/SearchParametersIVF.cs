using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Search.Parameters;

public sealed class SearchParametersIVF : SearchParameters
{
    public SearchParametersIVF(int nprobe, int maxCodes = 0, IDSelector? selector = null) 
        : base(CreateHandle(nprobe, maxCodes))
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
}