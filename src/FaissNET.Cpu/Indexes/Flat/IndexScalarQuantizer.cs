using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Flat;

internal readonly struct IndexScalarQuantizerRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexScalarQuantizer_free(handle);
}

/// <summary>
/// Flat (exhaustive) index that compresses each vector component independently with a scalar quantizer.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public sealed class IndexScalarQuantizer : CpuFlatFloatIndex<IndexScalarQuantizer>, IFromNativeIndexHandle<IndexScalarQuantizer>, IGpuClonableIndex<IndexScalarQuantizer, GpuIndexFlat>, ITrainableFloatIndex, IFlatIndex
{
    public readonly ScalarQuantizer ScalarQuantizer;

    /// <param name="dimensions">The number of dimensions for vectors in this index.</param>
    /// <exception cref="FaissException">Thrown when the index creation fails.</exception>
    public IndexScalarQuantizer(int dimensions, QuantizerType qt = QuantizerType.QT_8bit, MetricType metric = MetricType.L2) : this(CreateHandle(dimensions, qt, metric))
    {
    }


    private IndexScalarQuantizer(FaissIndexHandle handle) : base(handle)
    {
        FaissScalarQuantizerHandle scalarQuantizerHandle = new FaissScalarQuantizerHandle(Native.faiss_IndexScalarQuantizer_sq(NativeHandle));
        ScalarQuantizer = new ScalarQuantizer(scalarQuantizerHandle);
    }

    private static FaissIndexHandle CreateHandle(int dimensions, QuantizerType qt = QuantizerType.QT_8bit, MetricType metric = MetricType.L2)
    {
        if (metric != MetricType.L2 && metric != MetricType.InnerProduct)
        {
            throw new ArgumentOutOfRangeException(nameof(metric), "Metric must be L2 or InnerProduct");
        }

        FaissErrorHandler.ThrowIfError(Native.faiss_IndexScalarQuantizer_new_with(out IntPtr handle, dimensions, qt, metric));
        return new FaissIndexHandle<IndexScalarQuantizerRelease>(handle);
    }

    static IndexScalarQuantizer IFromNativeIndexHandle<IndexScalarQuantizer>.FromHandle(FaissIndexHandle handle) => new(handle);

    /// <inheritdoc />
    public bool IsTrained => ((ITrainableIndex)this).IsTrained;

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    /// <inheritdoc />
    public override void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }
        
        base.Add(count, vectors);
    }

    bool IGpuClonableIndex<IndexScalarQuantizer, GpuIndexFlat>.IsGpuClonable() => ScalarQuantizer.QuantizerType == QuantizerType.QT_fp16;

    public override void Dispose()
    {
        ScalarQuantizer.Dispose();
        base.Dispose();
    }
}

public class ScalarQuantizer
{
    private FaissScalarQuantizerHandle _handle;

    public ScalarQuantizer(FaissScalarQuantizerHandle handle)
    {
        _handle = handle;
    }

    internal void Dispose()
    {
        _handle.SetHandleAsInvalid();
    }

    public QuantizerType QuantizerType => Native.faiss_ScalarQuantizer_qtype(_handle);

    /// <summary>
    /// Bits per scalar code
    /// </summary>
    public long Bits => (long)Native.faiss_ScalarQuantizer_bits(_handle);

    /// <summary>
    /// Dimension of the input vectors
    /// </summary>
    public long Dimension => (long)Native.faiss_ScalarQuantizer_d(_handle);

    /// <summary>
    /// Bytes per encoded vector
    /// </summary>
    public long CodeSize => (long)Native.faiss_ScalarQuantizer_code_size(_handle);

    /// <summary>
    /// Range estimation strategy (uniform encoder)
    /// </summary>
    public RangeStat RangeStat => Native.faiss_ScalarQuantizer_rangestat(_handle);

    /// <summary>
    /// Argument to the range estimation strategy (rs)
    /// </summary>
    public float RangeStatArg => Native.faiss_ScalarQuantizer_rangestat_arg(_handle);

    /// <summary>
    /// Number of trained values
    /// </summary>
    public nuint TrainedSize => Native.faiss_ScalarQuantizer_trained_size(_handle);

    /// <summary>
    /// Copy the trained values into <paramref name="trainedValues"/>, which must hold at least <see cref="TrainedSize"/> floats.
    /// </summary>
    public void CopyTrainedValues(Span<float> trainedValues)
    {
        if (trainedValues.Length != (int)TrainedSize)
        {
            throw new ArgumentOutOfRangeException(nameof(trainedValues), $"TrainedValues must be of the same length as {nameof(TrainedSize)}");
        }

        Native.faiss_ScalarQuantizer_trained(_handle, trainedValues);
    }
}