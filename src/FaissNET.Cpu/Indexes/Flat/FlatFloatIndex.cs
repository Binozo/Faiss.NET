using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

/// <inheritdoc cref="FlatFloatIndex{T}" />
public abstract class FlatFloatIndex<T> : FloatIndex, IFlatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IComputeResidualFloatIndex where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    internal FlatFloatIndex(FaissIndexHandle handle) : base(handle) { }
    
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public virtual float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public virtual float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);
}

/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public abstract class CpuFlatFloatIndex<T> : FlatFloatIndex<T>, IIDSequentialFloatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, ICodeFloatIndex, IClonableFloatIndex<T>, ICpuFloatIndex where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    internal CpuFlatFloatIndex(FaissIndexHandle handle) : base(handle) { }

    public virtual void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);
    
    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);
    
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes)  => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);
    
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors)  => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);
    
    public T Clone() => ClonableFloatIndexImpl<T>.Clone(this);
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public abstract class GpuFlatFloatIndex<T> : FlatFloatIndex<T> where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    internal GpuFlatFloatIndex(FaissIndexHandle handle) : base(handle) { }
}