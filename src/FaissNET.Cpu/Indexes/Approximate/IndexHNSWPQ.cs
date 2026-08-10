using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// HNSW + Product Quantization.
/// </summary>
public sealed class IndexHNSWPQ : IndexHNSW, ITrainableFloatIndex, IClonableFloatIndex<IndexHNSWPQ>, IFromNativeIndexHandle<IndexHNSWPQ>
{
    public IndexHNSWPQ(int dimensions, int m = 32, int productQuantization = 16, int? subQuantizer = null, bool polysemy = false) : this($"HNSW{m},PQ{productQuantization}{(subQuantizer != null ? $"x{subQuantizer}" : string.Empty)}{(polysemy ? string.Empty : "np")}", dimensions, MetricType.L2)
    {
    }
    
    internal IndexHNSWPQ(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    internal IndexHNSWPQ(FaissIndexHandle handle) : base(handle)
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

    static IndexHNSWPQ IFromNativeIndexHandle<IndexHNSWPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexHNSWPQ>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexHNSWPQ Clone() => ((IClonableFloatIndex<IndexHNSWPQ>)this).Clone();
}