using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

public sealed class FaissIndexBinaryHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissIndexBinaryHandle() : base(true)
    {
    }

    internal FaissIndexBinaryHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_IndexBinary_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}