using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

/// <summary>
/// Inverted file index with flat (uncompressed) storage in inverted lists.
/// The quantizer partitions vectors into clusters; at search time only
/// the closest <see cref="Nprobe"/> clusters are scanned.
/// </summary>
public sealed class IndexIVFFlat : CpuIndex<IndexIVFFlat>, IFromNativeHandle<IndexIVFFlat>, IIVFIndex
{
    private readonly INativeIndex _quantizer;

    public IndexIVFFlat(INativeIndex quantizer, int dimensions, int nlist, MetricType metric = MetricType.L2)
    {
        _quantizer = quantizer;
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIVFFlat_new_with_metric(
                out IntPtr ptr,
                quantizer.Handle,
                (UIntPtr)dimensions,
                (UIntPtr)nlist,
                metric)
        );

        SafeHandle = new FaissIndexHandle(ptr);
        
        Native.faiss_IndexIVFFlat_set_own_fields(SafeHandle, false);
    }

    private IndexIVFFlat(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexIVFFlat_set_own_fields(SafeHandle, true);
    }

    static IndexIVFFlat IFromNativeHandle<IndexIVFFlat>.FromHandle(IntPtr handle) => new(handle);

    /// <summary>Number of inverted lists (clusters).</summary>
    public int Nlist => (int)Native.faiss_IndexIVFFlat_nlist(SafeHandle);

    /// <summary>Number of clusters to probe at search time.</summary>
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVFFlat_nprobe(SafeHandle);
        set => Native.faiss_IndexIVFFlat_set_nprobe(SafeHandle, (UIntPtr)value);
    }

    /// <summary>
    /// In-place update of vectors. The index must have a direct map.
    /// Call <see cref="MakeDirectMap"/> first if not already done.
    /// </summary>
    public unsafe void UpdateVectors(ReadOnlySpan<long> ids, ReadOnlySpan<float> vectors)
    {
        if (ids.Length != vectors.Length / Dimensions)
            throw new ArgumentException("ID count must match vector count.");
        fixed (long* pIdx = ids)
        fixed (float* pV = vectors)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexIVFFlat_update_vectors(
                    SafeHandle,
                    ids.Length,
                    pIdx,
                    pV)
            );
        }
    }
    
    /// <inheritdoc />
    public void MakeDirectMap(bool maintainDirectMap)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIVF_make_direct_map(SafeHandle, maintainDirectMap)
        );
    }

    /// <inheritdoc />
    public double ImbalanceFactor => Native.faiss_IndexIVF_imbalance_factor(SafeHandle);
}