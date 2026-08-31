using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public abstract class IDSelector : IDisposable
{
    internal FaissIDSelectorHandle SafeHandle { get; }
    
    protected internal IDSelector(IntPtr handle)
    {
        SafeHandle = new FaissIDSelectorHandle(handle);
    }
    
    /// <summary>Include IDs in the range [imin, imax).</summary>
    public static IDSelector Range(long imin, long imax) => new IDSelectorRange(imin, imax);

    /// <summary>Include IDs from a set.</summary>
    public static IDSelector Batch(params long[] indices) => new IDSelectorBatch(indices);

    /// <summary>Include IDs from a span.</summary>
    public static IDSelector Batch(ReadOnlySpan<long> indices) => new IDSelectorBatch(indices);

    /// <summary>Include IDs where the corresponding bitmap bit is set.</summary>
    public static IDSelector Bitmap(byte[] bitmap) => new IDSelectorBitmap(bitmap);
    
    /// <summary>Selector whose membership is the negation of this selector.</summary>
    /// <remarks>
    /// The result borrows this instance. Do not dispose this selector while the result is in use.
    /// <c>using var sel = IDSelector.Batch(1, 2).Not();</c> is safe: the Batch is rooted by the result.
    /// <c>using (var b = IDSelector.Batch(1, 2)) { n = b.Not(); }</c> then using <c>n</c> after the block is not.
    /// </remarks>
    public IDSelector Not() => new IDSelectorNot(this);

    /// <summary>Selector whose membership is the intersection of this selector and <paramref name="other"/>.</summary>
    /// <param name="other">The other operand. Must outlive the returned selector.</param>
    /// <remarks>
    /// The result borrows both operands. Do not dispose either while the result is in use.
    /// </remarks>
    public IDSelector And(IDSelector other) => new IDSelectorAnd(this, other);

    /// <summary>Selector whose membership is the union of this selector and <paramref name="other"/>.</summary>
    /// <param name="other">The other operand. Must outlive the returned selector.</param>
    /// <remarks>
    /// The result borrows both operands. Do not dispose either while the result is in use.
    /// </remarks>
    public IDSelector Or(IDSelector other) => new IDSelectorOr(this, other);

    /// <summary>Selector whose membership is the symmetric difference of this selector and <paramref name="other"/>.</summary>
    /// <param name="other">The other operand. Must outlive the returned selector.</param>
    /// <remarks>
    /// The result borrows both operands. Do not dispose either while the result is in use.
    /// </remarks>
    public IDSelector XOr(IDSelector other) => new IDSelectorXOr(this, other);
    
    /// <summary>
    /// Check if a vector with the given id is represented by this selector.
    /// </summary>
    /// <param name="id">Vector id</param>
    /// <returns></returns>
    public bool IsMember(long id) => Native.faiss_IDSelector_is_member(SafeHandle, id) != 0;
    
    public virtual void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}