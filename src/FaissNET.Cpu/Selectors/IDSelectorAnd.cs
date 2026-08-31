using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorAnd : IDSelector
{
    private readonly IDSelector _lhs;
    private readonly IDSelector _rhs;
    
    /// <summary>Intersection of <paramref name="lhs"/> and <paramref name="rhs"/>.</summary>
    /// <param name="lhs">Left operand. Must remain undisposed for the lifetime of this instance.</param>
    /// <param name="rhs">Right operand. Must remain undisposed for the lifetime of this instance.</param>
    /// <exception cref="ArgumentNullException"><paramref name="lhs"/> or <paramref name="rhs"/> is <see langword="null"/>.</exception>
    public IDSelectorAnd(IDSelector lhs, IDSelector rhs) : base(CreateHandle(lhs, rhs))
    {
        _lhs = lhs;
        _rhs = rhs;
    }
    
    private static IntPtr CreateHandle(IDSelector lhs, IDSelector rhs)
    {
        ArgumentNullException.ThrowIfNull(lhs);
        ArgumentNullException.ThrowIfNull(rhs);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IDSelectorAnd_new(out IntPtr ptr, lhs.SafeHandle, rhs.SafeHandle)
        );

        return ptr;
    }
}