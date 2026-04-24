using Faiss.Cpu.Selectors;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Search.Parameters;

public class SearchParameters : INativeSearchParameters
{
    internal FaissSearchParametersHandle SafeHandle { get; }
    
    IntPtr INativeSearchParameters.DangerousGetHandle() => SafeHandle.DangerousGetHandle();
    
    public SearchParameters(IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParameters_new(out IntPtr ptr, selector?.SafeHandle.DangerousGetHandle() ?? IntPtr.Zero)
        );
    
        SafeHandle = new FaissSearchParametersHandle(ptr);
    }

    internal SearchParameters(IntPtr handle)
    {
        SafeHandle = new FaissSearchParametersHandle(handle);
    }

    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}