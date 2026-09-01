using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Search.Parameters;

public class SearchParameters : IDisposable
{
    internal FaissSearchParametersHandle SafeHandle { get; }
    private readonly IDSelector? _selector;
    
    /// <summary>Optional per-query search parameters. <paramref name="selector"/> restricts kNN to matching ids.</summary>
    /// <param name="selector">
    /// Ids to consider during search, or <see langword="null"/> for no filter.
    /// Must remain undisposed for the lifetime of this instance. Native SearchParameters does not own the selector.
    /// </param>
    /// <remarks>
    /// This object borrows <paramref name="selector"/>. Do not dispose the selector while these parameters are in use.
    /// <c>using var sp = new SearchParameters(IDSelector.Batch(1, 2).Not());</c> is safe: the selector is rooted by this instance.
    /// </remarks>
    public SearchParameters(IDSelector? selector = null) :  this(CreateHandle(selector), selector)
    {
    }

    internal SearchParameters(FaissSearchParametersHandle handle, IDSelector? selector)
    {
        SafeHandle = handle;
        _selector = selector;
    }

    private static FaissSearchParametersHandle CreateHandle(IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParameters_new(out IntPtr ptr, selector?.SafeHandle)
        );

        return new FaissSearchParametersHandle(ptr);
    }

    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}