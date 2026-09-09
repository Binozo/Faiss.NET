using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHNSW : BinaryIndex, IIDSequentialBinaryIndex, IParamsBinarySearchIndex, IReconstructBinaryIndex, ICpuBinaryIndex, ISerializableBinaryIndex, IClonableBinaryIndex<IndexBinaryHNSW>, IFromNativeBinaryIndexHandle<IndexBinaryHNSW>
{
    public IndexBinaryHNSW(int dimensions, int m = 32) : this(CreateHandle(dimensions, m))
    {
    }

    private IndexBinaryHNSW(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    public void Add(long count, ReadOnlySpan<byte> vectors) => IDSequentialBinaryIndexImpl.Add(this, count, vectors);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels) => ParamsBinarySearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public byte[] Reconstruct(long key) => ReconstructBinaryIndexImpl.Reconstruct(this, key);

    public byte[] Reconstruct(long startKey, long count) => ReconstructBinaryIndexImpl.Reconstruct(this, startKey, count);
    
    private static FaissBinaryIndexHandle CreateHandle(int dimensions, int m)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, $"BHNSW{m}"));
        return new FaissBinaryIndexHandle(ptr);
    }

    static IndexBinaryHNSW IFromNativeBinaryIndexHandle<IndexBinaryHNSW>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);
    
    public IndexBinaryHNSW Clone() => ClonableBinaryIndexImpl<IndexBinaryHNSW>.Clone(this);
}