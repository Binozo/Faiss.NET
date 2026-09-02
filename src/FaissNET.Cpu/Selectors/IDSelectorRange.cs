using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorRange : IDSelector
{
    public IDSelectorRange(long imin, long imax) : base(CreateHandle(imin, imax)) { }
    
    public long Imin => Native.faiss_IDSelectorRange_imin(SafeHandle);

    public long Imax => Native.faiss_IDSelectorRange_imax(SafeHandle);
    
    private static IntPtr CreateHandle(long imin, long imax)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IDSelectorRange_new(out IntPtr ptr, imin, imax)
        );
        
        return ptr;
    }
}