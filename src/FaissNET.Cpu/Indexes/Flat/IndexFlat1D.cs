using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

/// <summary>
/// Specialized 1D flat index with sorted permutation and binary search.
/// Orders of magnitude faster than a generic flat index for single-dimensional data.
/// </summary>
public sealed class IndexFlat1D : CpuIndex<IndexFlat1D>, IFromNativeHandle<IndexFlat1D>, IFlatIndex
{
    public IndexFlat1D(bool continuousUpdate = true)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexFlat1D_new_with(out IntPtr ptr, continuousUpdate)
        );

        SafeHandle = new FaissIndexHandle(ptr);
    }

    private IndexFlat1D(IntPtr handle) : base(handle)
    {
        
    }

    static IndexFlat1D IFromNativeHandle<IndexFlat1D>.FromHandle(IntPtr handle) => new(handle);

    /// <summary>
    /// Manually rebuilds the sorted permutation of the database.
    /// Only needed when <see cref="ContinuousUpdate"/> is false.
    /// </summary>
    public void UpdatePermutation()
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexFlat1D_update_permutation(SafeHandle)
        );
    }
}