using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

/// <summary>
/// Binary inverted file index with coarse quantization and binary flat lists.
/// </summary>
public sealed class IndexBinaryIVF : CpuBinaryIndex<IndexBinaryIVF>, IFromNativeBinaryHandle<IndexBinaryIVF>
{
    /// <summary>
    /// Creates a binary IVF index.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality in bits.</param>
    /// <param name="nlist">Number of inverted lists (coarse clusters).</param>
    public IndexBinaryIVF(int dimensions, int nlist)
    {
        string description = $"BIVF{nlist}";

        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, description)
        );

        SafeHandle = new FaissIndexBinaryHandle(ptr);
    }
    
    /// <summary>
    /// Creates a binary IVF index with an HNSW quantizer.
    /// </summary>
    public static IndexBinaryIVF WithHNSWQuantizer(int dimensions, int nlist, int m = 32)
    {
        string description = $"BIVF{nlist}_HNSW{m}";
        return BinaryIndexFactory.Create<IndexBinaryIVF>(description, dimensions);
    }

    private IndexBinaryIVF(IntPtr handle) : base(handle)
    {
    }

    static IndexBinaryIVF IFromNativeBinaryHandle<IndexBinaryIVF>.FromHandle(IntPtr handle) => new(handle);

    public int Nlist => (int)Native.faiss_IndexBinaryIVF_nlist(SafeHandle);

    public int Nprobe
    {
        get => (int)Native.faiss_IndexBinaryIVF_nprobe(SafeHandle);
        set => Native.faiss_IndexBinaryIVF_set_nprobe(SafeHandle, (UIntPtr)value);
    }

    public int MaxCodes
    {
        get => (int)Native.faiss_IndexBinaryIVF_max_codes(SafeHandle);
        set => Native.faiss_IndexBinaryIVF_set_max_codes(SafeHandle, (UIntPtr)value);
    }

    public bool UseHeap
    {
        get => Native.faiss_IndexBinaryIVF_use_heap(SafeHandle) != 0;
        set => Native.faiss_IndexBinaryIVF_set_use_heap(SafeHandle, value);
    }

    public bool PerInvlistSearch
    {
        get => Native.faiss_IndexBinaryIVF_per_invlist_search(SafeHandle) != 0;
        set => Native.faiss_IndexBinaryIVF_set_per_invlist_search(SafeHandle, value);
    }

    public double ImbalanceFactor =>
        Native.faiss_IndexBinaryIVF_imbalance_factor(SafeHandle);

    public void MakeDirectMap(bool maintainDirectMap)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexBinaryIVF_make_direct_map(SafeHandle, maintainDirectMap)
        );
    }

    public void PrintStats() => Native.faiss_IndexBinaryIVF_print_stats(SafeHandle);
}