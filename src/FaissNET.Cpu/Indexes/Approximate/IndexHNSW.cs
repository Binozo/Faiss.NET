using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// Hierarchical Navigable Small World index.
/// Industry-standard graph-based approximate nearest neighbor search.
/// </summary>
public class IndexHNSW : FloatIndex, IFlatIndex, IRangeSearchFloatIndex, IIDSequentialFloatIndex, IParamsFloatSearchIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, IClonableFloatIndex<IndexHNSW>, IFromNativeIndexHandle<IndexHNSW>
{
    /// <summary>
    /// Creates an HNSW index with flat (exact) storage.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality</param>
    /// <param name="m">Number of neighbors per graph node</param>
    /// <param name="metricType">Distance metric</param>
    public IndexHNSW(int dimensions, int m = 32, MetricType metricType = MetricType.L2) : this($"HNSW{dimensions}", m, metricType)
    {
    }

    protected IndexHNSW(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
        
    }

    internal IndexHNSW(FaissIndexHandle handle) : base(handle)
    {
    }

    static IndexHNSW IFromNativeIndexHandle<IndexHNSW>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexHNSW>(description, dimensions, metricType).NativeHandle;
    }

    public virtual void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialFloatIndex)this).Add(count, vectors);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersHNSW parameters, Span<float> distances, Span<long> labels) =>
        ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);
    
    public float[] Reconstruct(long key) =>  ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count)  => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    public virtual IndexHNSW Clone() => ((IClonableFloatIndex<IndexHNSW>)this).Clone();
}