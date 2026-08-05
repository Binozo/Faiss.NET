using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Exceptions;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

internal readonly struct IndexIVFScalarQuantizerRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexIVFScalarQuantizer_free(handle);
}

public sealed class IndexIVFScalarQuantizer<T> : CpuFlatFloatIndex<IndexIVFScalarQuantizer<T>>, IIVFIndex, ITrainableFloatIndex, IIDMappedFloatIndex, IFromNativeIndexHandle<IndexIVFScalarQuantizer<T>>,
    IGpuClonableIndex<IndexIVFScalarQuantizer<T>, GpuIndexIVFScalarQuantizer<T>> where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    public readonly ScalarQuantizer ScalarQuantizer;
    private readonly T _quantizer;

    public IndexIVFScalarQuantizer(T quantizer, int dimensions, int nlist, QuantizerType qt, MetricType metric = MetricType.L2, bool encodeResidual = true, bool ownQuantizer = false) : this(
        CreateHandle(quantizer, dimensions, nlist, qt, metric, encodeResidual), quantizer, ownQuantizer)
    {
    }

    private IndexIVFScalarQuantizer(FaissIndexHandle handle, T? quantizer = null, bool ownFields = false) : base(handle)
    {
        OwnQuantizer = ownFields;
        _quantizer = quantizer ?? T.FromPointer(Native.faiss_IndexIVFScalarQuantizer_quantizer(NativeHandle));

        FaissScalarQuantizerHandle scalarQuantizerHandle = new FaissScalarQuantizerHandle(Native.faiss_IndexIVFScalarQuantizer_sq(NativeHandle));
        ScalarQuantizer = new ScalarQuantizer(scalarQuantizerHandle);
    }

    private static FaissIndexHandle CreateHandle(T quantizer, int dimensions, int nlist, QuantizerType qt, MetricType metric = MetricType.L2, bool encodeResidual = true)
    {
        if (quantizer.Dimensions != dimensions)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), $"Dimensions must match {nameof(quantizer)}.Dimensions");
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFScalarQuantizer_new_with_metric(out IntPtr handle, quantizer.Handle, (nuint)dimensions, (nuint)nlist, qt, metric, encodeResidual));
        return new FaissIndexHandle<IndexIVFScalarQuantizerRelease>(handle);
    }

    static IndexIVFScalarQuantizer<T> IFromNativeIndexHandle<IndexIVFScalarQuantizer<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public bool OwnQuantizer
    {
        get => Native.faiss_IndexIVFScalarQuantizer_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIVFScalarQuantizer_set_own_fields(NativeHandle, value);
    }

    /// <inheritdoc />
    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    /// <inheritdoc />
    public int Nlist => (int)Native.faiss_IndexIVFScalarQuantizer_nlist(NativeHandle);

    /// <inheritdoc />
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVFScalarQuantizer_nprobe(NativeHandle);
        set => Native.faiss_IndexIVFScalarQuantizer_set_nprobe(NativeHandle, (nuint)value);
    }

    /// <inheritdoc />
    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained && ScalarQuantizer.QuantizerType != QuantizerType.QT_0bit)
        {
            throw new FaissUntrainedException();
        }

        base.Add(count, vectors);
    }

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained && ScalarQuantizer.QuantizerType != QuantizerType.QT_0bit)
        {
            throw new FaissUntrainedException();
        }

        ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    }

    /// <inheritdoc cref="IParamsFloatSearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (ISearchParameters)parameters, distances, labels);

    /// <inheritdoc />
    public void MakeDirectMap(bool maintainDirectMap) => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVF_make_direct_map(NativeHandle, maintainDirectMap));

    public override float[] Reconstruct(long key)
    {
        MakeDirectMap(true);
        return base.Reconstruct(key);
    }

    public override float[] Reconstruct(long startKey, long count)
    {
        MakeDirectMap(true);
        return base.Reconstruct(startKey, count);
    }

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);

    bool IGpuClonableIndex<IndexIVFScalarQuantizer<T>, GpuIndexIVFScalarQuantizer<T>>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct && ScalarQuantizer.QuantizerType is QuantizerType.QT_8bit
        or QuantizerType.QT_8bit_uniform or QuantizerType.QT_8bit_direct
        or QuantizerType.QT_4bit or QuantizerType.QT_4bit_uniform or QuantizerType.QT_6bit or QuantizerType.QT_fp16;

    public override void Dispose()
    {
        ScalarQuantizer.Dispose();
        base.Dispose();
    }
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexIVFScalarQuantizer<T> : FloatIndex, ITrainableFloatIndex, IIDMappedFloatIndex, IIDSequentialFloatIndex, IParamsFloatSearchIndex, IFromNativeIndexHandle<GpuIndexIVFScalarQuantizer<T>>,
    IGpuIndex<IndexIVFScalarQuantizer<T>> where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    public GpuIndexIVFScalarQuantizer(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFScalarQuantizer<T> IFromNativeIndexHandle<GpuIndexIVFScalarQuantizer<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    /// <inheritdoc />
    public bool IsTrained => ((ITrainableFloatIndex)this).IsTrained;

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDMappedFloatIndex)this).Add(count, vectors, xids);
    }

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }

    /// <inheritdoc />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, ISearchParameters parameters, Span<float> distances, Span<long> labels) =>
        ((IParamsFloatSearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);
}