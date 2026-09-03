using Faiss.Cpu.Search.Parameters;
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
    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels);
}

internal static class ParamsBinarySearchIndexImpl
{
    public static void SearchWithParams(INativeBinaryIndex index, long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels)
    {
        unsafe
        {
            fixed (byte* pQuery = queryVectors)
            fixed (int* pDistances = distances)
            fixed (long* pLabels = labels)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_IndexBinary_search_with_params(index.Handle, count, pQuery, k, parameters.SafeHandle, pDistances, pLabels)
                );
            }
        }
    }
}