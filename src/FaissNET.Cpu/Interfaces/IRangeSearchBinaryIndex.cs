using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IRangeSearchBinaryIndex : INativeBinaryIndex, IBinaryIndex
{
    public unsafe void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result)
    {
        fixed (byte* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_range_search(Handle, count, pQuery, radius, result.SafeHandle) // TODO: Verify result
            );
        }
    }
}