using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorNot : IDSelector
{
    private readonly IDSelector _sel;
    
    /// <summary>Negation of <paramref name="sel"/>.</summary>
    /// <param name="sel">Operand. Must remain undisposed for the lifetime of this instance.</param>
    /// <exception cref="ArgumentNullException"><paramref name="sel"/> is <see langword="null"/>.</exception>
    public IDSelectorNot(IDSelector sel) : base(CreateHandle(sel))
    {
        _sel = sel;
    }
    
    private static IntPtr CreateHandle(IDSelector sel)
    {
        ArgumentNullException.ThrowIfNull(sel);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IDSelectorNot_new(out IntPtr ptr, sel.SafeHandle)
        );

        return ptr;
    }
}