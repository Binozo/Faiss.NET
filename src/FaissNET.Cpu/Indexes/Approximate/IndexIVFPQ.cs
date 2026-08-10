using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Exceptions;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// IVF + Product Quantization index.
/// </summary>
public sealed class IndexIVFPQ : FloatIndex, IIVFIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ITrainableFloatIndex, ICpuFloatIndex, IGpuClonableIndex<IndexIVFPQ, GpuIndexIVFPQ>, IClonableFloatIndex<IndexIVFPQ>, IFromNativeIndexHandle<IndexIVFPQ>
{
    private readonly IndexFlat _quantizer;

    public IndexIVFPQ(int dimensions, int n = 4096, int m = 16, int? subQuantizer = null, bool polysemy = false, MetricType metricType = MetricType.L2) : this(CreateHandle($"IVF{n},PQ{m}{(subQuantizer != null ? $"x{subQuantizer}" : string.Empty)}{(polysemy ? string.Empty : "np")}", dimensions, CheckMetricType(metricType)))
    {
    }

    internal IndexIVFPQ(FaissIndexHandle handle) : base(handle)
    {
        _quantizer = new IndexFlat(new FaissIndexHandle(Native.faiss_IndexIVF_quantizer(handle), false));
    }

    private static MetricType CheckMetricType(MetricType metricType)
    {
        if (metricType != MetricType.L2 && metricType != MetricType.InnerProduct)
        {
            throw new ArgumentException($"Unsupported metric type: {metricType}");
        }
        
        return metricType;
    }

    public bool OwnQuantizer
    {
        get => Native.faiss_IndexIVF_own_fields(NativeHandle) != 0;
        private set =>  Native.faiss_IndexIVF_set_own_fields(NativeHandle, value);
    }

    /// <inheritdoc />
    public int Nlist => (int)Native.faiss_IndexIVF_nlist(NativeHandle);

    /// <inheritdoc />
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVF_nprobe(NativeHandle);
        set => Native.faiss_IndexIVF_set_nprobe(NativeHandle, (nuint)value);
    }

    public QuantizerTrainMode QuantizerTrainMode => (QuantizerTrainMode)Native.faiss_IndexIVF_quantizer_trains_alone(NativeHandle);

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);

    public long RemoveIds(IIDSelector selector)
    {
        MakeDirectMap(true);
        
        return ((IIDRemovableFloatIndex)this).RemoveIds(selector);
    }

    public float[] Reconstruct(long key)
    {
        MakeDirectMap(true);

        return ((IReconstructFloatIndex)this).Reconstruct(key);
    }

    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    /// <inheritdoc />
    public void MakeDirectMap(bool maintainDirectMap) => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVF_make_direct_map(NativeHandle, maintainDirectMap));

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexIVFRelease>(handle, ownsHandle);

    static IndexIVFPQ IFromNativeIndexHandle<IndexIVFPQ>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexIVFPQ IFromNativeIndexHandle<IndexIVFPQ>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType)
    {
        return new FaissIndexHandle<IndexIVFRelease>(IndexFactory.Create<IndexIVFPQ>(description, dimensions, metricType).NativeHandle.DangerousGetHandle());
    }

    public IndexIVFPQ Clone() => ((IClonableFloatIndex<IndexIVFPQ>)this).Clone();

    bool IGpuClonableIndex<IndexIVFPQ, GpuIndexIVFPQ>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct;
}

public class GpuIndexIVFPQ : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IGpuIndex<IndexIVFPQ>, IFromNativeIndexHandle<GpuIndexIVFPQ>
{
    private GpuIndexIVFPQ(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFPQ IFromNativeIndexHandle<GpuIndexIVFPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) => ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);
}