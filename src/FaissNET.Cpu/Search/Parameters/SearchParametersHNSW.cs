using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Search.Parameters;

internal readonly struct SearchParametersHNSWRelease : IFaissRelease // TODO: Verify free by downcasting to SearchParameters
{
    public static void Release(IntPtr handle) => Native.faiss_SearchParametersHNSW_free(handle);
}

public sealed class SearchParametersHNSW : SearchParameters
{
    /// <summary>Per-query HNSW search parameters.</summary>
    /// <param name="efSearch">Search frontier size.</param>
    /// <param name="selector">
    /// Ids to consider during search, or <see langword="null"/> for no filter.
    /// Must remain undisposed for the lifetime of this instance.
    /// </param>
    /// <remarks>
    /// This object borrows <paramref name="selector"/>. Do not dispose the selector while these parameters are in use.
    /// Float HNSW also accepts a base <see cref="SearchParameters"/> for <c>sel</c> only; Binary HNSW requires this type.
    /// </remarks>
    public SearchParametersHNSW(int efSearch, IDSelector? selector = null)
        : base(CreateHandle(efSearch, selector), selector)
    {
    }
    
    private static FaissSearchParametersHandle CreateHandle(int efSearch, IDSelector? selector = null)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_SearchParametersHNSW_new_with(out IntPtr ptr, selector?.SafeHandle, efSearch)
        );

        return new FaissSearchParametersHandle<SearchParametersHNSWRelease>(ptr);
    }
    
    public int EfSearch
    {
        get => Native.faiss_SearchParametersHNSW_efSearch(SafeHandle);
        set => Native.faiss_SearchParametersHNSW_set_efSearch(SafeHandle, value);
    }
    
    public bool CheckRelativeDistance
    {
        get => Native.faiss_SearchParametersHNSW_check_relative_distance(SafeHandle) != 0;
        set => Native.faiss_SearchParametersHNSW_set_check_relative_distance(SafeHandle, value);
    }
    
    public bool BoundedQueue
    {
        get => Native.faiss_SearchParametersHNSW_bounded_queue(SafeHandle) != 0;
        set => Native.faiss_SearchParametersHNSW_set_bounded_queue(SafeHandle, value);
    }
}