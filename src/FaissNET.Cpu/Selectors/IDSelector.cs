using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public abstract class IDSelector : IDisposable
{
    internal FaissIDSelectorHandle SafeHandle { get; }
    
    protected IDSelector(IntPtr handle)
    {
        SafeHandle = new FaissIDSelectorHandle(handle);
    }
    
    public bool IsMember(long id) => Native.faiss_IDSelector_is_member(SafeHandle, id) != 0;
    
    public IDSelector Not() => new IDSelectorNot(this);
    public IDSelector And(IDSelector other) => new IDSelectorAnd(this, other);
    public IDSelector Or(IDSelector other) => new IDSelectorOr(this, other);
    public IDSelector XOr(IDSelector other) => new IDSelectorXOr(this, other);
    
    /// <summary>Include IDs in the range [imin, imax).</summary>
    public static IDSelector Range(long imin, long imax) => new IDSelectorRange(imin, imax);
    /// <summary>Include IDs from a set.</summary>
    public static IDSelector Batch(params long[] indices) => new IDSelectorBatch(indices);
    /// <summary>Include IDs from a span.</summary>
    public static IDSelector Batch(ReadOnlySpan<long> indices) => new IDSelectorBatch(indices);
    /// <summary>Include IDs where the corresponding bitmap bit is set.</summary>
    public static IDSelector Bitmap(ReadOnlySpan<byte> bitmap) => new IDSelectorBitmap(bitmap);
    
    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}