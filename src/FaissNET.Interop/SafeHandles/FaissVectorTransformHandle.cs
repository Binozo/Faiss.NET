using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

internal sealed class FaissVectorTransformHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissVectorTransformHandle() : base(true)
    {
    }

    internal FaissVectorTransformHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_VectorTransform_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}