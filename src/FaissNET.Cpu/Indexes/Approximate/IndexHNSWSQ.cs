using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// HNSW + Scalar Quantization. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexHNSWSQ : IndexHNSW, ITrainableFloatIndex, IClonableFloatIndex<IndexHNSWSQ>, IFromNativeIndexHandle<IndexHNSWSQ>
{
    public IndexHNSWSQ(int dimensions, QuantizerType quantizerType = QuantizerType.QT_8bit, int m = 32, MetricType metricType = MetricType.L2) : this(CreateDescription(quantizerType, m), dimensions, CheckMetricType(metricType))
    {
    }
    
    internal IndexHNSWSQ(string description, int dimensions, MetricType metricType) : this(CreateHandle(description, dimensions, metricType))
    {
    }

    internal IndexHNSWSQ(FaissIndexHandle handle) : base(handle)
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

    private static string CreateDescription(QuantizerType quantizerType, int m)
    {
        string qtStr = quantizerType switch
        {
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

        return $"HNSW{m},SQ{qtStr}";
    }

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        base.Add(count, vectors);
    }

    static IndexHNSWSQ IFromNativeIndexHandle<IndexHNSWSQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return IndexFactory.Create<IndexHNSWSQ>(description, dimensions, metricType).NativeHandle;
    }

    public override IndexHNSWSQ Clone() => ((IClonableFloatIndex<IndexHNSWSQ>)this).Clone();
}