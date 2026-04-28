using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

internal sealed class FaissClusteringHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissClusteringHandle() : base(true)
    {
    }

    internal FaissClusteringHandle(IntPtr preexistingHandle) : base(true) => SetHandle(preexistingHandle);

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_Clustering_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}