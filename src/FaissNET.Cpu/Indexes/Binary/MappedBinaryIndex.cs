using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Search;
using ITrainableBinaryIndex = Faiss.Cpu.Interfaces.ITrainableBinaryIndex;

namespace Faiss.Cpu.Indexes.Binary;

/// <inheritdoc cref="MappedBinaryIndex{T, TIndex}"/>
public class MappedBinaryIndex<T, TIndex> : BinaryIndex, IRangeSearchBinaryIndex, IIDRemovableBinaryIndex, IIDMappedBinaryIndex, ITrainableBinaryIndex, ICpuBinaryIndex, IParamsBinarySearchIndex, IClonableBinaryIndex<T> where T : MappedBinaryIndex<T, TIndex>, IFromNativeBinaryIndexHandle<T> where TIndex : IIDSequentialBinaryIndex, IBinaryIndex, IFromNativeBinaryIndexHandle<TIndex>
{
    protected MappedBinaryIndex(FaissBinaryIndexHandle handle) : base(handle) {}
    
    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => ((IIDMappedBinaryIndex)this).Add(count, vectors, xids);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => ((IRangeSearchBinaryIndex)this).RangeSearch(count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, ISearchParameters parameters, Span<int> distances, Span<long> labels) => ((IParamsBinarySearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableBinaryIndex)this).RemoveIds(selector);

    public new bool IsTrained => ((ITrainableBinaryIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) =>  ((ITrainableBinaryIndex)this).TrainAsync(count, vectors);

    public T Clone() => ((IClonableBinaryIndex<T>)this).Clone();
}