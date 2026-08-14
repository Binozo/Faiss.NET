using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IRangeSearchFloatIndex : INativeIndex, IFloatIndex
{
    public unsafe void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result)
    {
        fixed (float* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_range_search(Handle, count, pQuery, radius, result.SafeHandle)
            );
        }
    }
}