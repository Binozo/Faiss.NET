using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

public sealed class IndexNSGFlat : IndexNSG, IClonableFloatIndex<IndexNSGFlat>, IFromNativeIndexHandle<IndexNSGFlat>
{
    public IndexNSGFlat(int dimensions, int r = 32, MetricType metricType = MetricType.L2) : base(dimensions, "Flat", r, metricType)
    {
    }

    internal IndexNSGFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    static IndexNSGFlat IFromNativeIndexHandle<IndexNSGFlat>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexNSGFlat>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexNSGFlat Clone() => ((IClonableFloatIndex<IndexNSGFlat>)this).Clone();
}