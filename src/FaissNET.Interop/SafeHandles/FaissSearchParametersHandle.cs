using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Cpu.Search;

internal sealed class FaissSearchParametersHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissSearchParametersHandle() : base(true)
    {
        
    }
    
    internal FaissSearchParametersHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }
    
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_SearchParameters_free(handle);
            handle = IntPtr.Zero;
            
        }
        return true;
    }

}