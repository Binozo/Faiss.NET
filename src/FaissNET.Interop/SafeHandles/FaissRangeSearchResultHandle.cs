using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

internal sealed class FaissRangeSearchResultHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissRangeSearchResultHandle() : base(true)
    {
        
    }
    
    internal FaissRangeSearchResultHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }
    
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            NativeMethods.Native.faiss_RangeSearchResult_free(handle);
            handle = IntPtr.Zero;
        }
    
        return true;
    }
}