using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Factory;

public class GenericFloatIndex : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IRangeSearchFloatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IIDRemovableFloatIndex, ICodeFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IClonableFloatIndex<GenericFloatIndex>, IFromNativeIndexHandle<GenericFloatIndex>
{
    internal GenericFloatIndex(FaissIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => IDMappedFloatIndexImpl.Add(this, count, vectors, xids);

    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    static GenericFloatIndex IFromNativeIndexHandle<GenericFloatIndex>.FromHandle(FaissIndexHandle handle) => new(handle);

    public GenericFloatIndex Clone() => ClonableFloatIndexImpl<GenericFloatIndex>.Clone(this);
}

public class GpuGenericFloatIndex : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IGpuIndex<GenericFloatIndex>, IFromNativeIndexHandle<GpuGenericFloatIndex>
{
    private GpuGenericFloatIndex(FaissIndexHandle handle) : base(handle)
    {
    }
    
    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids) => IDMappedFloatIndexImpl.Add(this, count, vectors, xids);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) => ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    static GpuGenericFloatIndex IFromNativeIndexHandle<GpuGenericFloatIndex>.FromHandle(FaissIndexHandle handle) => new(handle);
}