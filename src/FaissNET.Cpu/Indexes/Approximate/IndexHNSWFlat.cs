using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <inheritdoc/>
public class IndexHNSWFlat : IndexHNSW, IFlatIndex, IClonableFloatIndex<IndexHNSWFlat>, IFromNativeIndexHandle<IndexHNSWFlat>
{
    public IndexHNSWFlat(int dimensions, int m = 32, MetricType metricType = MetricType.L2) : this($"HNSW{m},Flat", dimensions, metricType)
    {
    }
    
    protected IndexHNSWFlat(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    internal IndexHNSWFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    static IndexHNSWFlat IFromNativeIndexHandle<IndexHNSWFlat>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexHNSWFlat>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexHNSWFlat Clone() => ((IClonableFloatIndex<IndexHNSWFlat>)this).Clone();
}