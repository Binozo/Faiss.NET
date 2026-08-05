using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Binary;

/// <summary>
/// Exact binary flat index. Performs exhaustive Hamming search on packed binary vectors.
/// </summary>
public sealed class IndexBinaryFlat : BinaryIndex, IFromNativeBinaryIndexHandle<IndexBinaryFlat>, IRangeSearchBinaryIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex, IClonableBinaryIndex<IndexBinaryFlat>
{
    /// <summary>
    /// Creates an exact binary flat index.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality in bits.</param>
    public IndexBinaryFlat(int dimensions) : this(CreateHandle(dimensions))
    {
    }

    private IndexBinaryFlat(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    private static FaissBinaryIndexHandle CreateHandle(int dimensions)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, "BFlat"));
        return new FaissBinaryIndexHandle(ptr);
    }

    static IndexBinaryFlat IFromNativeBinaryIndexHandle<IndexBinaryFlat>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => ((IRangeSearchBinaryIndex)this).RangeSearch(count, queryVectors, radius, result);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableBinaryIndex)this).RemoveIds(selector);
    
    public byte[] Reconstruct(long key) =>  ((IReconstructBinaryIndex)this).Reconstruct(key);

    public byte[] Reconstruct(long startKey, long count)  => ((IReconstructBinaryIndex)this).Reconstruct(startKey, count);

    public IndexBinaryFlat Clone() => ((IClonableBinaryIndex<IndexBinaryFlat>)this).Clone();
}