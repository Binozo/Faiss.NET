using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Factory;

public sealed class GenericBinaryIndex : BinaryIndex, ITrainableBinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IRangeSearchBinaryIndex, IParamsBinarySearchIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex, ICpuBinaryIndex, ISerializableBinaryIndex, IFromNativeBinaryIndexHandle<GenericBinaryIndex>
{
    internal GenericBinaryIndex(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => ((ITrainableBinaryIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) => ((ITrainableBinaryIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors) => ((IIDSequentialBinaryIndex)this).Add(count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => ((IIDMappedBinaryIndex)this).Add(count, vectors, xids);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableBinaryIndex)this).RemoveIds(selector);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => ((IRangeSearchBinaryIndex)this).RangeSearch(count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, ISearchParameters parameters, Span<int> distances, Span<long> labels) => ((IParamsBinarySearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public byte[] Reconstruct(long key) => ((IReconstructBinaryIndex)this).Reconstruct(key);

    public byte[] Reconstruct(long startKey, long count) => ((IReconstructBinaryIndex)this).Reconstruct(startKey, count);

    static GenericBinaryIndex IFromNativeBinaryIndexHandle<GenericBinaryIndex>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);
}