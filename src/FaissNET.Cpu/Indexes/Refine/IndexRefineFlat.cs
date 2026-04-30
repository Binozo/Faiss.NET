using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Refine;

/// <summary>
/// Wraps a fast approximate index and re-ranks results using an exact flat index.
/// </summary>
public sealed class IndexRefineFlat : CpuIndex<IndexRefineFlat>, IFromNativeHandle<IndexRefineFlat>
{
    private readonly INativeIndex _baseIndex;

    public IndexRefineFlat(INativeIndex index)
    {
        _baseIndex = index;

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexRefineFlat_new(out IntPtr ptr, _baseIndex.Handle)
        );

        SafeHandle = new FaissIndexHandle(ptr);
        Native.faiss_IndexRefineFlat_set_own_fields(SafeHandle, false);
    }
    
    internal IndexRefineFlat(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexRefineFlat_set_own_fields(SafeHandle, true);
    }

    static IndexRefineFlat IFromNativeHandle<IndexRefineFlat>.FromHandle(IntPtr handle) => new(handle);
    
    /// <summary>
    /// Oversampling factor for refinement. Must be >= 1.
    /// When searching for k results, the base index is queried for k * KFactor candidates,
    /// which are then re-ranked exactly by the internal flat index.
    /// </summary>
    public float KFactor
    {
        get => Native.faiss_IndexRefineFlat_k_factor(SafeHandle);
        set
        {
            if (value < 1.0f)
                throw new ArgumentOutOfRangeException(nameof(value), "KFactor must be >= 1");
            Native.faiss_IndexRefineFlat_set_k_factor(SafeHandle, value);
        }
    }
}