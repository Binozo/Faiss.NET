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

    internal IndexNSG(FaissIndexHandle handle) : base(handle)
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

    public virtual void Add(long count, ReadOnlySpan<float> vectors) => IDSequentialFloatIndexImpl.Add(this, count, vectors);

    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType) => IndexFactory.Create<IndexNSG>(description, dimensions, metricType).NativeHandle;

    static IndexNSG IFromNativeIndexHandle<IndexNSG>.FromHandle(FaissIndexHandle handle) => new(handle);

    public virtual IndexNSG Clone() => ClonableFloatIndexImpl<IndexNSG>.Clone(this);
}