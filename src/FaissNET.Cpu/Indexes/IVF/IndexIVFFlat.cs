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
    /// <param name="quantizer">The coarse quantizer.</param>
    /// <param name="ownQuantizer">
    /// If <c>true</c>, this index frees the quantizer and <paramref name="quantizer"/> is closed immediately.
    /// If <c>false</c>, the caller keeps ownership. This index keeps the quantizer alive until it is freed, even if the caller disposes it first.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="quantizer"/> is part of another index, such as <c>IndexIDMap.SubIndex</c>.</exception>
    public IndexIVFFlat(T quantizer, int dimensions, int nlist, MetricType metric = MetricType.L2, bool ownQuantizer = false)
        : this(CreateHandle(OwnershipGuard.RequireIndependent(quantizer, nameof(quantizer)), dimensions, nlist, metric, ownQuantizer), ownQuantizer)
    {
        if (ownQuantizer)
        {
            quantizer.Handle.SetHandleAsInvalid();
        }
    }

    /// <param name="ownFields">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    private IndexIVFFlat(FaissIndexHandle handle, bool? ownFields = null) : base(handle)
    {
        if (ownFields is bool own)
        {
            OwnQuantizer = own;
        }
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

    /// <remarks>If enabling fails, the index is left without a direct map.</remarks>
    public bool DirectMap
    {
        set => FloatDirectMapImpl.Make(NativeHandle, value);
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
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        var dimensions = Dimensions;

        if (vectors.Length % dimensions != 0)
            throw new ArgumentOutOfRangeException(nameof(vectors), $"{nameof(vectors)}.Length must be divisible by {nameof(Dimensions)}");
        if (vectors.Length / dimensions != xids.Length)
            throw new ArgumentOutOfRangeException(nameof(xids), $"xids must match {nameof(vectors)}.Length / {nameof(Dimensions)}");
        if (xids.Length != precomputedIdx.Length)
            throw new ArgumentOutOfRangeException(nameof(precomputedIdx), $"precomputedIdx must match {nameof(xids)}.Length");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_add_core(NativeHandle, vectors.Length / dimensions, vectors, xids, precomputedIdx));
    }

    /// <inheritdoc cref="IParamsFloatSearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);

    private protected override KeyLayout KeyLayout => KeyLayout.CheckedIds;

    /// <summary>
    /// In-place update of vectors. The index must have a direct map.
    /// Set <see cref="DirectMap"/> first.
    /// </summary>
    public unsafe void UpdateVectors(ReadOnlySpan<long> ids, ReadOnlySpan<float> vectors)
    {
        if (vectors.Length != (long)ids.Length * Dimensions)
            throw new ArgumentException($"Expected {ids.Length} vectors of {Dimensions} floats ({(long)ids.Length * Dimensions}), got {vectors.Length} floats.", nameof(vectors));

        fixed (long* pIdx = ids)
        fixed (float* pV = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_update_vectors(NativeHandle, ids.Length, pIdx, pV));
        }
    }

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);
    
    private static FaissIndexHandle CreateHandle(T quantizer, int dimensions, int nlist, MetricType metric, bool ownQuantizer)
    {
        if (quantizer.Dimensions != dimensions)
            throw new ArgumentOutOfRangeException(nameof(dimensions), $"Dimensions must match {nameof(quantizer)}.Dimensions");

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFFlat_new_with_metric(out IntPtr handle, quantizer.Handle, (nuint)dimensions, (nuint)nlist, metric));

        if (ownQuantizer)
            return new FaissIndexHandle<IndexIVFFlatRelease>(handle);

        var keepAlive = new FaissDependentIndexHandle<IndexIVFFlatRelease>(handle);
        keepAlive.AddDependency(quantizer.Handle);
        return keepAlive;
    }

    static IndexIVFFlat<T> IFromNativeIndexHandle<IndexIVFFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexIVFFlatRelease>(handle, ownsHandle);

    static IndexIVFFlat<T> IFromNativeIndexHandle<IndexIVFFlat<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    bool IGpuClonableIndex<IndexIVFFlat<T>, GpuIndexIVFFlat<T>>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct;
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexIVFFlat<T> : GpuFlatFloatIndex<GpuIndexIVFFlat<T>>, ITrainableFloatIndex, IIDMappedFloatIndex, IIDSequentialFloatIndex, IFromNativeIndexHandle<GpuIndexIVFFlat<T>>, IGpuIndex<IndexIVFFlat<T>>, IGpuIVFIndex where T: class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    private GpuIndexIVFFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFFlat<T> IFromNativeIndexHandle<GpuIndexIVFFlat<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    bool IGpuIVFIndex.Uses32BitIds { get; set; }

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

        GpuIVFIDs.ThrowIfNot32Bit(this, xids);
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