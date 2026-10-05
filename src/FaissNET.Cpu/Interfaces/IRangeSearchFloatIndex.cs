using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IRangeSearchFloatIndex : INativeIndex, IFloatIndex
{
    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result);
}

internal static class RangeSearchFloatIndexImpl
{
    public static unsafe void RangeSearch(INativeIndex index, long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long queryLength = checked(count * index.Dimensions);
        if (queryVectors.Length < queryLength)
            throw new ArgumentException($"Query span too small. Expected {queryLength}, got {queryVectors.Length}.", nameof(queryVectors));

        if (result.Nq < count)
            throw new ArgumentException($"Result was allocated for {result.Nq} queries, but {count} were requested.", nameof(result));

        fixed (float* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_range_search(index.Handle, count, pQuery, radius, result.SafeHandle)
            );
        }
    }
}