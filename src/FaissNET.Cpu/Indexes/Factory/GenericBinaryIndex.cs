using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Factory;

public sealed class GenericBinaryIndex : BinaryIndex, ITrainableBinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IRangeSearchBinaryIndex, IParamsBinarySearchIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex, ICpuBinaryIndex, ISerializableBinaryIndex, IFromNativeBinaryIndexHandle<GenericBinaryIndex>
{
    internal GenericBinaryIndex(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => TrainableBinaryIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) => TrainableBinaryIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors) => IDSequentialBinaryIndexImpl.Add(this, count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => IDMappedBinaryIndexImpl.Add(this, count, vectors, xids);

    public long RemoveIds(IDSelector selector) => IDRemovableBinaryIndexImpl.RemoveIds(this, selector);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => RangeSearchBinaryIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels) => ParamsBinarySearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public byte[] Reconstruct(long key) => ReconstructBinaryIndexImpl.Reconstruct(this, key);

    public byte[] Reconstruct(long startKey, long count) => ReconstructBinaryIndexImpl.Reconstruct(this, startKey, count);

    static GenericBinaryIndex IFromNativeBinaryIndexHandle<GenericBinaryIndex>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);
}