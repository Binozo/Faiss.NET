using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Flat;

/// <summary>
/// Base flat index.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public sealed class IndexFlat : CpuFlatFloatIndex<IndexFlat>, IFromNativeIndexHandle<IndexFlat>, IGpuClonableIndex<IndexFlat, GpuIndexFlat>
{
    public IndexFlat() : base(CreateHandle())
    {
    }

    public IndexFlat(long dimensions, MetricType metric) : base(CreateHandle(dimensions, metric))
    {
    }

    internal IndexFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    private static FaissIndexHandle CreateHandle()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlat_new(out var handle));
        return new FaissIndexHandle(handle);
    }

    private static FaissIndexHandle CreateHandle(long dimensions, MetricType metric)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlat_new_with(out var handle, dimensions, metric));
        return new FaissIndexHandle(handle);
    }

    static IndexFlat IFromNativeIndexHandle<IndexFlat>.FromHandle(FaissIndexHandle handle) => new(handle);

    bool IGpuClonableIndex<IndexFlat, GpuIndexFlat>.IsGpuClonable() => Metric is MetricType.L2 or MetricType.InnerProduct;
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexFlat : GpuFlatFloatIndex<GpuIndexFlat>, IFromNativeIndexHandle<GpuIndexFlat>, IGpuIndex<IndexFlat>
{
    private GpuIndexFlat(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexFlat IFromNativeIndexHandle<GpuIndexFlat>.FromHandle(FaissIndexHandle handle) => new(handle);
}