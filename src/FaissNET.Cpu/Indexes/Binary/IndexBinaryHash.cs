using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHash : BinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IParamsBinarySearchIndex, IRangeSearchBinaryIndex, IClonableBinaryIndex<IndexBinaryHash>, IFromNativeBinaryIndexHandle<IndexBinaryHash>
{
    public IndexBinaryHash(int dimensions, int leadingBits) : this(CreateHandle(dimensions, leadingBits))
    {
    }

    private IndexBinaryHash(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    private static FaissBinaryIndexHandle CreateHandle(int dimensions, int leadingBits)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, $"BHash{leadingBits}"));
        return new FaissBinaryIndexHandle(ptr);
    }

    static IndexBinaryHash IFromNativeBinaryIndexHandle<IndexBinaryHash>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

    public void Add(long count, ReadOnlySpan<byte> vectors) => ((IIDSequentialBinaryIndex)this).Add(count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => ((IIDMappedBinaryIndex)this).Add(count, vectors, xids);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => ((IRangeSearchBinaryIndex)this).RangeSearch(count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, ISearchParameters parameters, Span<int> distances, Span<long> labels) => ((IParamsBinarySearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public IndexBinaryHash Clone() => ((IClonableBinaryIndex<IndexBinaryHash>)this).Clone();
}