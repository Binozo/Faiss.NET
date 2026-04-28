using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

internal sealed class FaissParameterSpaceHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissParameterSpaceHandle() : base(true)
    {
    }

    internal FaissParameterSpaceHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_ParameterSpace_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}