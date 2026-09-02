using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// IVF + Product Quantization index.
/// </summary>
public sealed class IndexIVFPQ : FloatIndex, IIVFIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IRangeSearchFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IComputeResidualFloatIndex,
    ICodeFloatIndex, ITrainableFloatIndex, ICpuFloatIndex, ISerializableFloatIndex, IGpuClonableIndex<IndexIVFPQ, GpuIndexIVFPQ>, IClonableFloatIndex<IndexIVFPQ>, IFromNativeIndexHandle<IndexIVFPQ>
{
    private readonly IndexFlat _quantizer;

    public IndexIVFPQ(int dimensions, int n = 4096, int m = 16, int? subQuantizer = null, bool polysemy = false, MetricType metricType = MetricType.L2) : this(
        CreateHandle($"IVF{n},PQ{m}{(subQuantizer != null ? $"x{subQuantizer}" : string.Empty)}{(polysemy ? string.Empty : "np")}", dimensions, CheckMetricType(metricType)))
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
        private set => Native.faiss_IndexIVF_set_own_fields(NativeHandle, value);
    }

    public int Nlist => (int)Native.faiss_IndexIVF_nlist(NativeHandle);

    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVF_nprobe(NativeHandle);
        set => Native.faiss_IndexIVF_set_nprobe(NativeHandle, (nuint)value);
    }

    public bool DirectMap
    {
        set => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVF_make_direct_map(NativeHandle, value));
    }

    public QuantizerTrainMode QuantizerTrainMode => (QuantizerTrainMode)Native.faiss_IndexIVF_quantizer_trains_alone(NativeHandle);

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) =>
        ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);

    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);

    public long RemoveIds(IDSelector selector)
    {
        DirectMap = true;

        return IDRemovableFloatIndexImpl.RemoveIds(this, selector);
    }

    public float[] Reconstruct(long key)
    {
        DirectMap = true;

        return ReconstructFloatIndexImpl.Reconstruct(this, key);
    }

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);

    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);

    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);

    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);

    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);

    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);

    private static FaissIndexHandle CreateHandle(string description, int dimensions, MetricType metricType) => 
        new FaissIndexHandle<IndexIVFRelease>(IndexFactory.Create<IndexIVFPQ>(description, dimensions, metricType, ownsHandle: false).NativeHandle.DangerousGetHandle());

    static IndexIVFPQ IFromNativeIndexHandle<IndexIVFPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexIVFRelease>(handle, ownsHandle);

    static IndexIVFPQ IFromNativeIndexHandle<IndexIVFPQ>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    public IndexIVFPQ Clone() => ClonableFloatIndexImpl<IndexIVFPQ>.Clone(this);

    bool IGpuClonableIndex<IndexIVFPQ, GpuIndexIVFPQ>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct;
}

public class GpuIndexIVFPQ : FloatIndex, ITrainableFloatIndex, IIDSequentialFloatIndex, IIDMappedFloatIndex, IParamsFloatSearchIndex, IGpuIndex<IndexIVFPQ>, IFromNativeIndexHandle<GpuIndexIVFPQ>
{
    private GpuIndexIVFPQ(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFPQ IFromNativeIndexHandle<GpuIndexIVFPQ>.FromHandle(FaissIndexHandle handle) => new(handle);

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }

    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) =>
        ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);

    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);
}