using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

/// <summary>
/// Exact binary flat index. Performs exhaustive Hamming search on packed binary vectors.
/// </summary>
public sealed class IndexBinaryFlat : BinaryIndex, IIDSequentialBinaryIndex, IRangeSearchBinaryIndex, IParamsBinarySearchIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex, ICpuBinaryIndex, ISerializableBinaryIndex, IClonableBinaryIndex<IndexBinaryFlat>, IFromNativeBinaryIndexHandle<IndexBinaryFlat>
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

    public void Add(long count, ReadOnlySpan<byte> vectors) => IDSequentialBinaryIndexImpl.Add(this, count, vectors);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => RangeSearchBinaryIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels) => ParamsBinarySearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public long RemoveIds(IDSelector selector) => IDRemovableBinaryIndexImpl.RemoveIds(this, selector);
    
    public byte[] Reconstruct(long key) =>  ReconstructBinaryIndexImpl.Reconstruct(this, key);

    public byte[] Reconstruct(long startKey, long count)  => ReconstructBinaryIndexImpl.Reconstruct(this, startKey, count);

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

    public IndexBinaryFlat Clone() => ClonableBinaryIndexImpl<IndexBinaryFlat>.Clone(this);
}