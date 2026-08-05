using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Flat;

/// <inheritdoc cref="FlatFloatIndex{T}" />
public abstract class FlatFloatIndex<T> : FloatIndex, IFlatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    protected FlatFloatIndex(FaissIndexHandle handle) : base(handle) { }

    public virtual float[] Reconstruct(long key) =>  ((IReconstructFloatIndex)this).Reconstruct(key);

    public virtual float[] Reconstruct(long startKey, long count)  => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);
}

/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public abstract class CpuFlatFloatIndex<T> : FlatFloatIndex<T>, IRangeSearchFloatIndex, IIDRemovableFloatIndex, ICodeFloatIndex, IClonableFloatIndex<T>, ICpuFloatIndex, IIDSequentialIndex where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    protected CpuFlatFloatIndex(FaissIndexHandle handle) : base(handle) { }

    public virtual void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialIndex)this).Add(count, vectors);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);
    
    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();
    
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes)  => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);
    
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors)  => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);
    
    public T Clone() => ((IClonableFloatIndex<T>)this).Clone();
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public abstract class GpuFlatFloatIndex<T> : FlatFloatIndex<T> where T : FloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    protected GpuFlatFloatIndex(FaissIndexHandle handle) : base(handle) { }
}