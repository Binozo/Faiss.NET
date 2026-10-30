using System.Numerics;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

internal readonly struct IndexIVFScalarQuantizerRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexIVFScalarQuantizer_free(handle);
}

public sealed class IndexIVFScalarQuantizer<T> : CpuFlatFloatIndex<IndexIVFScalarQuantizer<T>>, IIVFIndex, ITrainableFloatIndex, IIDMappedFloatIndex, ISerializableFloatIndex, IFromNativeIndexHandle<IndexIVFScalarQuantizer<T>>,
    IGpuClonableIndex<IndexIVFScalarQuantizer<T>, GpuIndexIVFScalarQuantizer<T>>, IListNumberPrefixedCodes where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    public readonly ScalarQuantizer ScalarQuantizer;

    /// <inheritdoc cref="IndexIVFFlat{T}(T, int, int, MetricType, bool)"/>
    public IndexIVFScalarQuantizer(T quantizer, int dimensions, int nlist, QuantizerType qt, MetricType metric = MetricType.L2, bool encodeResidual = true, bool ownQuantizer = false) : this(
        CreateHandle(OwnershipGuard.RequireIndependent(quantizer, nameof(quantizer)), dimensions, nlist, qt, metric, encodeResidual, ownQuantizer), ownQuantizer)
    {
        if (ownQuantizer)
            quantizer.Handle.SetHandleAsInvalid();
    }

    /// <param name="ownFields">Override the ownership. Keep <see langword="null"/> when wrapping if you want to keep default faiss behavior.</param>
    private IndexIVFScalarQuantizer(FaissIndexHandle handle, bool? ownFields = null) : base(handle)
    {
        if (ownFields is bool own)
        {
            OwnQuantizer = own;
        }

        ScalarQuantizer = new ScalarQuantizer(new FaissBorrowedScalarQuantizerHandle(Native.faiss_IndexIVFScalarQuantizer_sq(handle), handle));
    }

    public bool OwnQuantizer
    {
        get => Native.faiss_IndexIVFScalarQuantizer_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexIVFScalarQuantizer_set_own_fields(NativeHandle, value);
    }

    /// <inheritdoc />
    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    /// <inheritdoc />
    public int Nlist => (int)Native.faiss_IndexIVFScalarQuantizer_nlist(NativeHandle);

    /// <inheritdoc />
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVFScalarQuantizer_nprobe(NativeHandle);
        set => Native.faiss_IndexIVFScalarQuantizer_set_nprobe(NativeHandle, (nuint)value);
    }

    /// <inheritdoc cref="IndexIVFFlat{T}.DirectMap"/>
    public bool DirectMap
    {
        set => FloatDirectMapImpl.Make(NativeHandle, value);
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

        IDMappedFloatIndexImpl.Add(this, count, vectors, xids);
    }

    /// <inheritdoc cref="IParamsFloatSearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParametersIVF parameters, Span<float> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);

    private protected override KeyLayout KeyLayout => KeyLayout.CheckedIds;

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(NativeHandle);

    private static FaissIndexHandle CreateHandle(T quantizer, int dimensions, int nlist, QuantizerType qt, MetricType metric, bool encodeResidual, bool ownQuantizer)
    {
        if (quantizer.Dimensions != dimensions)
        {
            throw new ArgumentOutOfRangeException(nameof(dimensions), $"Dimensions must match {nameof(quantizer)}.Dimensions");
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVFScalarQuantizer_new_with_metric(out IntPtr handle, quantizer.Handle, (nuint)dimensions, (nuint)nlist, qt, metric, encodeResidual));

        if (ownQuantizer)
            return new FaissIndexHandle<IndexIVFScalarQuantizerRelease>(handle);

        var keepAlive = new FaissDependentIndexHandle<IndexIVFScalarQuantizerRelease>(handle);
        keepAlive.AddDependency(quantizer.Handle);
        return keepAlive;
    }
    
    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexIVFScalarQuantizerRelease>(handle, ownsHandle);

    static IndexIVFScalarQuantizer<T> IFromNativeIndexHandle<IndexIVFScalarQuantizer<T>>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexIVFScalarQuantizer<T> IFromNativeIndexHandle<IndexIVFScalarQuantizer<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    bool IGpuClonableIndex<IndexIVFScalarQuantizer<T>, GpuIndexIVFScalarQuantizer<T>>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct && ScalarQuantizer.QuantizerType is QuantizerType.QT_8bit
        or QuantizerType.QT_8bit_uniform or QuantizerType.QT_8bit_direct
        or QuantizerType.QT_4bit or QuantizerType.QT_4bit_uniform or QuantizerType.QT_6bit or QuantizerType.QT_fp16;

    void IGpuClonableIndex<IndexIVFScalarQuantizer<T>, GpuIndexIVFScalarQuantizer<T>>.OnClonedToGpu(INativeIndex gpuIndex) =>
        ((GpuIndexIVFScalarQuantizer<T>)gpuIndex).QuantizerType = ScalarQuantizer.QuantizerType;

    public override void Dispose()
    {
        ScalarQuantizer.Dispose();
        base.Dispose();
    }
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
/// <remarks>
/// With a per-dimension quantizer (<see cref="QuantizerType.QT_8bit"/>, <see cref="QuantizerType.QT_4bit"/>,
/// <see cref="QuantizerType.QT_6bit"/>) the device scan keeps two floats per dimension next to its top-k queue in 48 KiB
/// of shared memory. Searches that do not fit (any k above 512; smaller k once the dimension grows into the thousands)
/// throw <see cref="ArgumentOutOfRangeException"/>: faiss would fail the kernel launch inside an assertion and abort.
/// </remarks>
public class GpuIndexIVFScalarQuantizer<T> : FloatIndex, ITrainableFloatIndex, IIDMappedFloatIndex, IIDSequentialFloatIndex, IParamsFloatSearchIndex, IFromNativeIndexHandle<GpuIndexIVFScalarQuantizer<T>>,
    IGpuIndex<IndexIVFScalarQuantizer<T>>, IGpuIVFIndex where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
{
    internal GpuIndexIVFScalarQuantizer(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexIVFScalarQuantizer<T> IFromNativeIndexHandle<GpuIndexIVFScalarQuantizer<T>>.FromHandle(FaissIndexHandle handle) => new(handle);

    bool IGpuIVFIndex.Uses32BitIds { get; set; }

    /// <summary>The quantizer of the CPU index this was transferred from.</summary>
    internal QuantizerType? QuantizerType { get; set; }

    internal override void ThrowIfKUnsupported(long k)
    {
        if (QuantizerType is not (Models.QuantizerType.QT_8bit or Models.QuantizerType.QT_4bit or Models.QuantizerType.QT_6bit))
            return;

        // ivfInterleavedScan: 128 threads (4 warps) each keep a queue of k rounded up to a power of two (at least 32)
        // of float distances and 64-bit ids in static shared memory; the codec adds vmin and vdiff per dimension.
        // Queues above 512 entries already fill the 48 KiB launch limit on their own.
        const long sharedMemoryLimit = 48 * 1024;
        var queue = Math.Max(32L, (long)BitOperations.RoundUpToPowerOf2((ulong)k));
        var required = 4 * queue * (sizeof(float) + sizeof(long)) + 2L * sizeof(float) * Dimensions;

        if (queue > 512 || required > sharedMemoryLimit)
            throw new ArgumentOutOfRangeException(nameof(k), k,
                $"A GPU {QuantizerType} scalar quantizer index with {Dimensions} dimensions cannot search for {k} neighbours: the device scan needs {required} bytes of shared memory, the limit is {sharedMemoryLimit}.");
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
    public void SearchWithParams(long count, ReadOnlySpan<float> queryVectors, int k, SearchParameters parameters, Span<float> distances, Span<long> labels) =>
        ParamsFloatSearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);
}