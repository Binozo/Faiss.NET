using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorNot : IDSelector
{
    private readonly IDSelector _sel;
    
    public IDSelectorNot(IDSelector sel) : base(CreateHandle(sel))
    {
        _sel = sel;
    }
    
    private static IntPtr CreateHandle(IDSelector sel)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IDSelectorNot_new(out IntPtr ptr, sel.SafeHandle)
        );

        return ptr;
    }
}