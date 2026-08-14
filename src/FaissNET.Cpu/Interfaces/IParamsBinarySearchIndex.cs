using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IParamsBinarySearchIndex : IBinaryIndex, INativeBinaryIndex
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
    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, ISearchParameters parameters, Span<int> distances, Span<long> labels)
    {
        unsafe
        {
            fixed (byte* pQuery = queryVectors)
            fixed (int* pDistances = distances)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError( // TODO: Improve this INativeSearchParameters handling
                    Native.faiss_IndexBinary_search_with_params(Handle, count, pQuery, k, ((INativeSearchParameters)parameters).Handle, pDistances, pLabels)
                );
            }
        }
    }
}