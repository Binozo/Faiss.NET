using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IRangeSearchBinaryIndex : INativeBinaryIndex, IBinaryIndex
{
    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result);
}

internal static class RangeSearchBinaryIndexImpl
{
    public static unsafe void RangeSearch(INativeBinaryIndex index, long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long queryLength = checked(count * index.CodeSize);
        if (queryVectors.Length < queryLength)
            throw new ArgumentException($"Query span too small. Expected {queryLength} bytes, got {queryVectors.Length}.", nameof(queryVectors));

        if (result.Nq < count)
            throw new ArgumentException($"Result was allocated for {result.Nq} queries, but {count} were requested.", nameof(result));

        fixed (byte* pQuery = queryVectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_range_search(index.Handle, count, pQuery, radius, result.SafeHandle)
            );
        }
    }
}