using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Search;

public readonly struct SearchParametersRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_SearchParameters_free(handle);
}

internal class FaissSearchParametersHandle : FaissHandle 
{
    public FaissSearchParametersHandle() : base(true)
    {
    }

    internal FaissSearchParametersHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle)
    {
    }
    
    protected override bool ReleaseHandle()
    {
        SearchParametersRelease.Release(handle);
        return true;
    }
}

internal class FaissSearchParametersHandle<T> : FaissSearchParametersHandle where T : struct, IFaissRelease
{
    public FaissSearchParametersHandle()
    { }

    internal FaissSearchParametersHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle) { }

    protected override bool ReleaseHandle()
    {
        T.Release(handle);
        return true;
    }
}