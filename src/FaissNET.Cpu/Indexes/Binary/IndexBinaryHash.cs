using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHash : BinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IRangeSearchBinaryIndex, ICpuBinaryIndex, ISerializableBinaryIndex, IFromNativeBinaryIndexHandle<IndexBinaryHash>
{
    public IndexBinaryHash(int dimensions, int leadingBits) : this(CreateHandle(dimensions, leadingBits))
    {
    }

    private IndexBinaryHash(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    public void Add(long count, ReadOnlySpan<byte> vectors) => IDSequentialBinaryIndexImpl.Add(this, count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => IDMappedBinaryIndexImpl.Add(this, count, vectors, xids);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => RangeSearchBinaryIndexImpl.RangeSearch(this, count, queryVectors, radius, result);


    private static FaissBinaryIndexHandle CreateHandle(int dimensions, int leadingBits)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        return BinaryIndexFactory.Create<IndexBinaryHash>($"BHash{leadingBits}", dimensions).NativeHandle;
    }

    static IndexBinaryHash IFromNativeBinaryIndexHandle<IndexBinaryHash>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

}