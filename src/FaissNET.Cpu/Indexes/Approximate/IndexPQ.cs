using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// Product Quantization flat index.
/// </summary>
public sealed class IndexPQ : FloatIndex, IFlatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, IClonableFloatIndex<IndexPQ>, IFromNativeIndexHandle<IndexPQ>
{
    public IndexPQ(int dimensions, int m = 16, MetricType metricType = MetricType.L2, int? nbits = null, bool polysemousTraining = true) : this($"PQ{m}{(nbits != null ? $"x{nbits}" : string.Empty)}{(polysemousTraining ? string.Empty : "np")}", dimensions, metricType)
    {
    }
    
    internal IndexPQ(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    internal IndexPQ(FaissIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }
        
        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);

    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    static IndexPQ IFromNativeIndexHandle<IndexPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexPQ>(description, dimensions, metricType).NativeHandle;
    }

    public IndexPQ Clone() => ((IClonableFloatIndex<IndexPQ>)this).Clone();
}