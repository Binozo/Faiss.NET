using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// Product Quantization flat index.
/// </summary>
public sealed class IndexPQ : FloatIndex, IFlatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IClonableFloatIndex<IndexPQ>, IFromNativeIndexHandle<IndexPQ>
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

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }
        
        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    static IndexPQ IFromNativeIndexHandle<IndexPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType) => IndexFactory.Create<IndexPQ>(description, dimensions, metricType).NativeHandle;

    public IndexPQ Clone() => ClonableFloatIndexImpl<IndexPQ>.Clone(this);
}