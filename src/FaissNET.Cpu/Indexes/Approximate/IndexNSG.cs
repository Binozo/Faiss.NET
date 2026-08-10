using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// NSG (Navigating Spreading-out Graph) index.
/// </summary>
public class IndexNSG : FloatIndex, IIDSequentialFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICpuFloatIndex, IClonableFloatIndex<IndexNSG>, IFromNativeIndexHandle<IndexNSG>
{
    internal IndexNSG(int dimensions, string description, int r = 32, MetricType metricType = MetricType.L2) : this($"NSG{r},{description}", dimensions, CheckMetricType(metricType))
    {
    }
    
    private IndexNSG(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    protected IndexNSG(FaissIndexHandle handle) : base(handle)
    {
    }

    private static MetricType CheckMetricType(MetricType metricType)
    {
        if (metricType != MetricType.L2 && metricType != MetricType.InnerProduct)
        {
            throw new ArgumentException($"Unsupported metric type: {metricType}");
        }
        
        return metricType;
    }

    public virtual void Add(long count, ReadOnlySpan<float> vectors) => ((IIDSequentialFloatIndex)this).Add(count, vectors);

    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    static IndexNSG IFromNativeIndexHandle<IndexNSG>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexNSG>(description, dimensions, metricType).NativeHandle;
    }

    public virtual IndexNSG Clone() => ((IClonableFloatIndex<IndexNSG>)this).Clone();
}