using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

public sealed class IndexNSGPQ : IndexNSG, ITrainableFloatIndex, IClonableFloatIndex<IndexNSGPQ>, IFromNativeIndexHandle<IndexNSGPQ>
{
    public IndexNSGPQ(int dimensions, int r = 32, int productQuantization = 16, int? subQuantizer = null, bool polysemy = false, MetricType metricType = MetricType.L2) : base(dimensions, $"PQ{productQuantization}{(subQuantizer != null ? $"x{subQuantizer}" : string.Empty)}{(polysemy ? string.Empty : "np")}", r, metricType)
    {
    }

    internal IndexNSGPQ(FaissIndexHandle handle) : base(handle)
    {
    }
    
    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        base.Add(count, vectors);
    }

    static IndexNSGPQ IFromNativeIndexHandle<IndexNSGPQ>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexNSGPQ>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexNSGPQ Clone() => ((IClonableFloatIndex<IndexNSGPQ>)this).Clone();
}