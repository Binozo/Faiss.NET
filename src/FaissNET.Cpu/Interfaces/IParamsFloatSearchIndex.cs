using Faiss.Interfaces;
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
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels)
    {
        unsafe
        {
            fixed (float* pQuery = queryVectors)
            fixed (float* pDistances = distances)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError( // TODO: Improve this INativeSearchParameters handling
                    Native.faiss_Index_search_with_params(Handle, count, pQuery, k, ((INativeSearchParameters)parameters).Handle, pDistances, pLabels)
                );
            }
        }
    }
}