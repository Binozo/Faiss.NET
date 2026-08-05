using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

internal readonly struct IndexFlat1DRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexFlat1D_free(handle);
}

/// <summary>
/// Specialized 1D flat index with sorted permutation and binary search.
/// Orders of magnitude faster than a generic flat index for single-dimensional data.
/// </summary>
/// <inheritdoc cref="CpuFlatFloatIndex{T}" />
public sealed class IndexFlat1D : CpuFlatFloatIndex<IndexFlat1D>, IFromNativeIndexHandle<IndexFlat1D>, IGpuClonableIndex<IndexFlat1D, GpuIndexFlat1D>
{
    public IndexFlat1D(bool continuousUpdate = true) : base(CreateHandle(continuousUpdate))
    {
    }

    private IndexFlat1D(FaissIndexHandle handle) : base(handle)
    {
        
    }

    private static FaissIndexHandle CreateHandle(bool continuousUpdate = true)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlat1D_new_with(out var handle, continuousUpdate));
        return new FaissIndexHandle<IndexFlat1DRelease>(handle);
    }

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexFlat1DRelease>(handle, ownsHandle);

    static IndexFlat1D IFromNativeIndexHandle<IndexFlat1D>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexFlat1D IFromNativeIndexHandle<IndexFlat1D>.FromHandle(FaissIndexHandle handle) => new(handle);

    /// <summary>
    /// Manually rebuilds the sorted permutation of the database.
    /// Only needed when <see cref="ContinuousUpdate"/> is false.
    /// </summary>
    public void UpdatePermutation() => FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlat1D_update_permutation(NativeHandle));
}

/// <inheritdoc cref="GpuFlatFloatIndex{T}" />
public class GpuIndexFlat1D : GpuFlatFloatIndex<GpuIndexFlat1D>, IFromNativeIndexHandle<GpuIndexFlat1D>, IGpuIndex<IndexFlat1D>
{
    private GpuIndexFlat1D(FaissIndexHandle handle) : base(handle)
    {
    }

    static GpuIndexFlat1D IFromNativeIndexHandle<GpuIndexFlat1D>.FromHandle(FaissIndexHandle handle) => new(handle);

    /// <summary>
    /// Manually rebuilds the sorted permutation of the database.
    /// Only needed when <see cref="ContinuousUpdate"/> is false.
    /// </summary>
    public void UpdatePermutation() => FaissErrorHandler.ThrowIfError(Native.faiss_IndexFlat1D_update_permutation(NativeHandle));
}