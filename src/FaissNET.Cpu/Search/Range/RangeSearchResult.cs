using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Search.Range;

/// <summary>
/// Variable-length results for range search (all vectors within a distance radius).
/// </summary>
public sealed class RangeSearchResult : IDisposable
{
    internal FaissRangeSearchResultHandle SafeHandle { get; }
    
    public RangeSearchResult(long queriesCount)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_RangeSearchResult_new(out IntPtr ptr, queriesCount)
        );
    
        SafeHandle = new FaissRangeSearchResultHandle(ptr);
    }
    
    public int Nq => (int)Native.faiss_RangeSearchResult_nq(SafeHandle);
    
    /// <summary>
    /// Get the result slice for a specific query.
    /// </summary>
    public unsafe RangeSearchQueryResult GetQueryResult(int queryIndex)
    {
        if (queryIndex < 0 || queryIndex >= Nq)
            throw new ArgumentOutOfRangeException(nameof(queryIndex));

        Native.faiss_RangeSearchResult_lims(SafeHandle, out IntPtr limsPtr);
        Native.faiss_RangeSearchResult_labels(SafeHandle, out IntPtr labelsPtr, out IntPtr distancesPtr);

        var lims = (long*)limsPtr;
        long start = lims[queryIndex];
        long end = lims[queryIndex + 1];
        int count = (int)(end - start);
        
        return new RangeSearchQueryResult
        {
            Labels = new ReadOnlySpan<long>((long*)labelsPtr + start, count),
            Distances = new ReadOnlySpan<float>((float*)distancesPtr + start, count)
        };
    }
    
    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }
}

public readonly ref struct RangeSearchQueryResult
{
    public ReadOnlySpan<long> Labels { get; init; }
    public ReadOnlySpan<float> Distances { get; init; }
}