using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

public sealed class IndexNSGSQ : IndexNSG, ITrainableFloatIndex, IClonableFloatIndex<IndexNSGSQ>, IFromNativeIndexHandle<IndexNSGSQ>
{
    public IndexNSGSQ(int dimensions, int r = 32, QuantizerType quantizerType = QuantizerType.QT_8bit, MetricType metricType = MetricType.L2) : base(dimensions, $"SQ{CreateDescription(quantizerType)}", r, metricType)
    {
    }

    internal IndexNSGSQ(FaissIndexHandle handle) : base(handle)
    {
    }
    
    private static string CreateDescription(QuantizerType quantizerType)
    {
        return quantizerType switch
        {
            QuantizerType.QT_0bit => "0",
            QuantizerType.QT_4bit => "4",
            QuantizerType.QT_6bit => "6",
            QuantizerType.QT_8bit => "8",
            QuantizerType.QT_8bit_direct => "8_direct",
            QuantizerType.QT_8bit_direct_signed => "8_direct_signed",
            QuantizerType.QT_fp16 => "fp16",
            QuantizerType.QT_bf16 => "bf16",
            QuantizerType.QT_1bit_tqmse => "tqmse1",
            QuantizerType.QT_2bit_tqmse => "tqmse2",
            QuantizerType.QT_3bit_tqmse => "tqmse3",
            QuantizerType.QT_4bit_tqmse => "tqmse4",
            QuantizerType.QT_8bit_tqmse => "tqmse8",
            QuantizerType.QT_2bit_tq => "tq2",
            QuantizerType.QT_3bit_tq => "tq3",
            QuantizerType.QT_4bit_tq => "tq4",
            QuantizerType.QT_5bit_tq => "tq5",
            _ => throw new ArgumentException($"Unsupported quantizer for HNSWSQ: {quantizerType}")
        };
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

    static IndexNSGSQ IFromNativeIndexHandle<IndexNSGSQ>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexNSGSQ>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexNSGSQ Clone() => ((IClonableFloatIndex<IndexNSGSQ>)this).Clone();
}