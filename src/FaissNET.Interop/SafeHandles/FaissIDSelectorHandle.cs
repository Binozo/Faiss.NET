using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Cpu.Selectors;

internal sealed class FaissIDSelectorHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissIDSelectorHandle() : base(true)
    {
        
    }
    
    internal FaissIDSelectorHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }
    
    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_IDSelector_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}