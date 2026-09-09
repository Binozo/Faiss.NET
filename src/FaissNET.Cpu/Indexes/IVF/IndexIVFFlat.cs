using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

internal readonly struct IndexIVFFlatRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexIVFFlat_free(handle);
}

public enum QuantizerTrainMode : sbyte
{
    /// <summary>
    /// Use the quantizer as index in a kmeans training
    /// </summary>
    QuantizerIndexKmeansTraining = 0,
    /// <summary>
    /// Just pass on the training set to the train() of the quantizer
    /// </summary>
    PassTrainingToQuantizer = 1,
    /// <summary>
    /// Kmeans training on a flat index + add the centroids to the quantizer
    /// </summary>
    TrainingOnFlatIndex = 2
}

/// <summary>
/// Inverted file index with flat (uncompressed) storage in inverted lists.
/// The quantizer partitions vectors into clusters; at search time only
/// the closest <see cref="Nprobe"/> clusters are scanned.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public sealed class IndexIVFFlat<T> : CpuFlatFloatIndex<IndexIVFFlat<T>>, IIVFIndex, ITrainableFloatIndex, IIDMappedFloatIndex, ISerializableFloatIndex, IFromNativeIndexHandle<IndexIVFFlat<T>>, IGpuClonableIndex<IndexIVFFlat<T>, GpuIndexIVFFlat<T>> where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    private readonly T _quantizer;

    public IndexIVFFlat(T quantizer, int dimensions, int nlist, MetricType metric = MetricType.L2, bool ownQuantizer = false) : this(CreateHandle(quantizer, dimensions, nlist, metric), quantizer, ownQuantizer)
    {
    }

    private IndexIVFFlat(FaissIndexHandle handle, T? quantizer = null, bool ownFields = false) : base(handle)
    {
        OwnQuantizer = ownFields;
        _quantizer = quantizer ?? T.FromPointer(Native.faiss_IndexIVFFlat_quantizer(handle));
    }

    public bool OwnQuantizer
    {
        get => Native.faiss_IndexIVFFlat_own_fields(NativeHandle) != 0;
        private set =>  Native.faiss_IndexIVFFlat_set_own_fields(NativeHandle, value);
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    /// <inheritdoc />
    public int Nlist => (int)Native.faiss_IndexIVFFlat_nlist(NativeHandle);

    /// <inheritdoc />
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVFFlat_nprobe(NativeHandle);
        set => Native.faiss_IndexIVFFlat_set_nprobe(NativeHandle, (nuint)value);
    }

    public bool DirectMap
    {
        set => FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVF_make_direct_map(NativeHandle, value));
    }

    public QuantizerTrainMode QuantizerTrainMode => (QuantizerTrainMode)Native.faiss_IndexIVFFlat_quantizer_trains_alone(NativeHandle);

    /// <inheritdoc />
    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        base.Add(count, vectors);
    }

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    public void AddCore(ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids, ReadOnlySpan<long> precomputedIdx)
    {
        if (vectors.Length != xids.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(vectors), $"vectors must match {nameof(vectors)}.Length");
        }

        if (xids.Length != precomputedIdx.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(xids), $"xids must match {nameof(precomputedIdx)}.Length");
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_add_core(NativeHandle, vectors.Length, vectors, xids, precomputedIdx));
    }

    /// <inheritdoc cref="IParamsFloatSearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);

    public override float[] Reconstruct(long key)
    {
        DirectMap = true;
        return base.Reconstruct(key);
    }

    public override float[] Reconstruct(long startKey, long count)
    {
        DirectMap = true;
        return base.Reconstruct(startKey, count);
    }

    /// <summary>
    /// In-place update of vectors. The index must have a direct map.
    /// Call <see cref="DirectMap"/> first if not already done.
    /// </summary>
    public unsafe void UpdateVectors(ReadOnlySpan<long> ids, ReadOnlySpan<float> vectors)
    {
        if (ids.Length != vectors.Length / Dimensions)
            throw new ArgumentException(nameof(ids), $"count must match {nameof(vectors)} count.");

        fixed (long* pIdx = ids)
        fixed (float* pV = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_update_vectors(NativeHandle, ids.Length, pIdx, pV));
        }
    }

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);
    
    private static FaissIndexHandle CreateHandle(T quantizer, int dimensions, int nlist, MetricType metric = MetricType.L2)
    {
        if (quantizer.Dimensions != dimensions)
            throw new ArgumentOutOfRangeException(nameof(dimensions), $"Dimensions must match {nameof(quantizer)}.Dimensions");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_new_with_metric(out IntPtr handle, quantizer.Handle, (nuint)dimensions, (nuint)nlist, metric));
        return new FaissIndexHandle<IndexIVFFlatRelease>(handle);
    }

    static IndexIVFFlat<T> IFromNativeIndexHandle<IndexIVFFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexIVFFlatRelease>(handle, ownsHandle);

    static IndexIVFFlat<T> IFromNativeIndexHandle<IndexIVFFlat<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    bool IGpuClonableIndex<IndexIVFFlat<T>, GpuIndexIVFFlat<T>>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct;
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexIVFFlat<T> : GpuFlatFloatIndex<GpuIndexIVFFlat<T>>, ITrainableFloatIndex, IIDMappedFloatIndex, IIDSequentialFloatIndex, IFromNativeIndexHandle<GpuIndexIVFFlat<T>>, IGpuIndex<IndexIVFFlat<T>> where T: class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    private GpuIndexIVFFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFFlat<T> IFromNativeIndexHandle<GpuIndexIVFFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }
}