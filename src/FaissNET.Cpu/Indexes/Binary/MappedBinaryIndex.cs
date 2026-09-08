using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

/// <inheritdoc cref="MappedBinaryIndex{T, TIndex}"/>
public class MappedBinaryIndex<T, TIndex> : BinaryIndex, IRangeSearchBinaryIndex, IIDRemovableBinaryIndex, IIDMappedBinaryIndex, ITrainableBinaryIndex, ICpuBinaryIndex, IParamsBinarySearchIndex, ISerializableBinaryIndex, IClonableBinaryIndex<T> where T : MappedBinaryIndex<T, TIndex>, IFromNativeBinaryIndexHandle<T> where TIndex : IIDSequentialBinaryIndex, IBinaryIndex, IFromNativeBinaryIndexHandle<TIndex>
{
    internal MappedBinaryIndex(FaissBinaryIndexHandle handle) : base(handle) {}

    public bool IsTrained => TrainableBinaryIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) =>  TrainableBinaryIndexImpl.TrainAsync(this, count, vectors);
    
    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => IDMappedBinaryIndexImpl.Add(this, count, vectors, xids);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => RangeSearchBinaryIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels) => ParamsBinarySearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public long RemoveIds(IDSelector selector) => IDRemovableBinaryIndexImpl.RemoveIds(this, selector);

    public T Clone() => ClonableBinaryIndexImpl<T>.Clone(this);
}