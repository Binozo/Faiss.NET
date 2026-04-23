using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorOr : IDSelector
{
    private readonly IDSelector _lhs;
    private readonly IDSelector _rhs;
    
    public IDSelectorOr(IDSelector lhs, IDSelector rhs) : base(CreateHandle(lhs, rhs))
    {
        _lhs = lhs;
        _rhs = rhs;
    }
    
    private static IntPtr CreateHandle(IDSelector lhs, IDSelector rhs)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IDSelectorOr_new(out IntPtr ptr, lhs.SafeHandle, rhs.SafeHandle)
        );

        return ptr;
    }
}