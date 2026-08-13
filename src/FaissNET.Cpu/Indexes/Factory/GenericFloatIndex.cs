using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Factory;

public class GenericFloatIndex : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IRangeSearchFloatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IIDRemovableFloatIndex, ICodeFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IClonableFloatIndex<GenericFloatIndex>, IFromNativeIndexHandle<GenericFloatIndex>
{
    internal GenericFloatIndex(FaissIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialFloatIndex)this).Add(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => ((IIDMappedFloatIndex)this).Add(count, vectors, xids);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    static GenericFloatIndex IFromNativeIndexHandle<GenericFloatIndex>.FromHandle(FaissIndexHandle handle) => new(handle);

    public GenericFloatIndex Clone() => ((IClonableFloatIndex<GenericFloatIndex>)this).Clone();
}

public class GpuGenericFloatIndex : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IGpuIndex<GenericFloatIndex>, IFromNativeIndexHandle<GpuGenericFloatIndex>
{
    private GpuGenericFloatIndex(FaissIndexHandle handle) : base(handle)
    {
    }
    
    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialFloatIndex)this).Add(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => ((IIDMappedFloatIndex)this).Add(count, vectors, xids);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    static GpuGenericFloatIndex IFromNativeIndexHandle<GpuGenericFloatIndex>.FromHandle(FaissIndexHandle handle) => new(handle);
}