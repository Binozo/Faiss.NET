using System.Runtime.InteropServices;

namespace Faiss.Interop.SafeHandles;

internal sealed partial class FaissIndexHandle : SafeHandle
{
    // Tells the base class that an IntPtr.Zero handle means it's invalid
    public FaissIndexHandle() : base(IntPtr.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            faiss_Index_free(handle);
        }
        return true;
    }

    [LibraryImport("faiss_c")]
    private static partial void faiss_Index_free(IntPtr obj);
}