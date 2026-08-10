using Faiss.Cpu.Interfaces;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHNSW : BinaryIndex, IIDSequentialBinaryIndex, IParamsBinarySearchIndex, IReconstructBinaryIndex, IClonableBinaryIndex<IndexBinaryHNSW>, IFromNativeBinaryIndexHandle<IndexBinaryHNSW>
{
    public IndexBinaryHNSW(int dimensions, int m = 32) : this(CreateHandle(dimensions, m))
    {
    }

    private IndexBinaryHNSW(FaissBinaryIndexHandle handle) : base(handle)
    {
    }
    
    private static FaissBinaryIndexHandle CreateHandle(int dimensions, int m)
    {
        if (dimensions <= 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, $"BHNSW{m}"));
        return new FaissBinaryIndexHandle(ptr);
    }

    public void Add(long count, ReadOnlySpan<byte> vectors) => ((IIDSequentialBinaryIndex)this).Add(count, vectors);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, ISearchParameters parameters, Span<int> distances, Span<long> labels) => ((IParamsBinarySearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public byte[] Reconstruct(long key) => ((IReconstructBinaryIndex)this).Reconstruct(key);

    public byte[] Reconstruct(long startKey, long count) => ((IReconstructBinaryIndex)this).Reconstruct(startKey, count);

    static IndexBinaryHNSW IFromNativeBinaryIndexHandle<IndexBinaryHNSW>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);
    
    public IndexBinaryHNSW Clone() => ((IClonableBinaryIndex<IndexBinaryHNSW>)this).Clone();
}