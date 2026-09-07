using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

public sealed class IndexNSGPQ : IndexNSG, ITrainableFloatIndex, IClonableFloatIndex<IndexNSGPQ>, IFromNativeIndexHandle<IndexNSGPQ>
{
    public IndexNSGPQ(int dimensions, int r = 32, int productQuantization = 16, int? subQuantizer = null, bool polysemy = false, MetricType metricType = MetricType.L2) : base(dimensions,
        $"PQ{productQuantization}{(subQuantizer != null ? $"x{subQuantizer}" : string.Empty)}{(polysemy ? string.Empty : "np")}", r, metricType)
    {
    }

    internal IndexNSGPQ(FaissIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        base.Add(count, vectors);
    }

    static IndexNSGPQ IFromNativeIndexHandle<IndexNSGPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    public override IndexNSGPQ Clone() => ClonableFloatIndexImpl<IndexNSGPQ>.Clone(this);
}