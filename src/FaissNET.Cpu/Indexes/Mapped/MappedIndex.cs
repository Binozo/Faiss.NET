using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Mapped;

/// <inheritdoc cref="MappedIndex{T, TIndex}"/>
public abstract class MappedIndex<T, TIndex> : FloatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IIDMappedFloatIndex, ITrainableFloatIndex, ICodeFloatIndex, ICpuFloatIndex, IParamsFloatSearchIndex, ISerializableFloatIndex, IClonableFloatIndex<T> where T : MappedIndex<T, TIndex>, IFromNativeIndexHandle<T> where TIndex : IIDSequentialFloatIndex, IFloatIndex, IFromNativeIndexHandle<TIndex>
{
    protected MappedIndex(FaissIndexHandle handle) : base(handle) { }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);
    
    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();
    
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes)  => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);
    
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors)  => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) =>  ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public T Clone() => ((IClonableFloatIndex<T>)this).Clone();
}