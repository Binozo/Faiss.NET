using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Mapped;

/// <inheritdoc cref="MappedIndex{T, TIndex}"/>
public abstract class MappedIndex<T, TIndex> : FloatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IIDMappedFloatIndex, ITrainableFloatIndex, ICodeFloatIndex, ICpuFloatIndex, IParamsFloatSearchIndex, ISerializableFloatIndex, IClonableFloatIndex<T> where T : MappedIndex<T, TIndex>, IFromNativeIndexHandle<T> where TIndex : IIDSequentialFloatIndex, IFloatIndex, IFromNativeIndexHandle<TIndex>
{
    internal MappedIndex(FaissIndexHandle handle) : base(handle) { }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);
    
    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);
    
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes)  => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);
    
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors)  => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public T Clone() => ClonableFloatIndexImpl<T>.Clone(this);
}