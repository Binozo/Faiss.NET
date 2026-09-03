using Faiss.Cpu.Search.Parameters;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IParamsFloatSearchIndex : IFloatIndex, INativeIndex
{
    /// <summary>
    /// Searches the index with the given params.
    /// </summary>
    /// <param name="count">Number of query vectors.</param>
    /// <param name="queryVectors">Query vectors (count * dimension)</param>
    /// <param name="k">Neighbors to return per query</param>
    /// <param name="parameters">Search options</param>
    /// <param name="distances">Out: count * k floats, needs to be allocated by user</param>
    /// <param name="labels">Out: count * k floats, needs to be allocated by user</param>
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels);
}

internal static class ParamsFloatSearchIndexImpl
{
    public static void SearchWithParams(INativeIndex index, long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels)
    {
        unsafe
        {
            fixed (float* pQuery = queryVectors)
            fixed (float* pDistances = distances)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_search_with_params(index.Handle, count, pQuery, k, parameters.SafeHandle, pDistances, pLabels)
                );
            }
        }
    }
}