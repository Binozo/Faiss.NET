using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

public sealed class IndexHNSWFlatPanorama : IndexHNSWFlat, IClonableFloatIndex<IndexHNSWFlatPanorama>, IFromNativeIndexHandle<IndexHNSWFlatPanorama>
{
    public IndexHNSWFlatPanorama(int dimensions, int m = 32, int numPanoramaLevels = 8) : this($"HNSW{m},FlatPanorama{numPanoramaLevels}", dimensions, MetricType.L2)
    {
    }
    
    internal IndexHNSWFlatPanorama(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    internal IndexHNSWFlatPanorama(FaissIndexHandle handle) : base(handle)
    {
    }

    static IndexHNSWFlatPanorama IFromNativeIndexHandle<IndexHNSWFlatPanorama>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexHNSWFlatPanorama>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexHNSWFlatPanorama Clone() => ((IClonableFloatIndex<IndexHNSWFlatPanorama>)this).Clone();
}