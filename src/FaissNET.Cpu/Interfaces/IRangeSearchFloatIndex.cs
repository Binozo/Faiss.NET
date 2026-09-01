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
        fixed (float* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_range_search(index.Handle, count, pQuery, radius, result.SafeHandle)
            );
        }
    }
}