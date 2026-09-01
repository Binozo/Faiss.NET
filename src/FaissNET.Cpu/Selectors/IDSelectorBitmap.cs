using System.Runtime.InteropServices;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Cpu.Selectors;

internal sealed class UnmanagedBitmapHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public UnmanagedBitmapHandle(IntPtr ptr) : base(true) => SetHandle(ptr);

    protected override unsafe bool ReleaseHandle()
    {
        NativeMemory.Free((void*)handle);
        handle = IntPtr.Zero;
        return true;
    }
}

public sealed class IDSelectorBitmap : IDSelector
{
    private readonly UnmanagedBitmapHandle _bitmap;

    public IDSelectorBitmap(byte[] bitmap) : this(Copy(bitmap)) { }

    private IDSelectorBitmap((IntPtr sel, UnmanagedBitmapHandle bitmap) t)
        : base(t.sel)
    {
        _bitmap = t.bitmap;
    }
    
    public int N => (int)Native.faiss_IDSelectorBitmap_n(SafeHandle);
    
    private static unsafe (IntPtr sel, UnmanagedBitmapHandle bitmap) Copy(byte[] bitmap)
    {
        ArgumentNullException.ThrowIfNull(bitmap);

        var dest = (byte*)NativeMemory.Alloc((nuint)bitmap.Length);
        var owned = new UnmanagedBitmapHandle((IntPtr)dest);
        try
        {
            bitmap.AsSpan().CopyTo(new Span<byte>(dest, bitmap.Length));

            FaissErrorHandler.ThrowIfError(
                Native.faiss_IDSelectorBitmap_new(
                    out IntPtr sel,
                    (UIntPtr)bitmap.Length,
                    dest));

            return (sel, owned);
        }
        catch
        {
            owned.Dispose();
            throw;
        }
    }
    
    public override void Dispose()
    {
        base.Dispose();
        _bitmap.Dispose();
    }
}