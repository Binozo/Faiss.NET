namespace Faiss.Interop.SafeHandles;

using Microsoft.Win32.SafeHandles;
using NativeMethods;

internal sealed partial class FaissIndexHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissIndexHandle() : base(true)
    {
    }
    
    internal FaissIndexHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_Index_free(handle);
            handle = IntPtr.Zero;
        }
        return true;
    }
}