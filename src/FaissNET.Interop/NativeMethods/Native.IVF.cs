using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVF_merge_from(FaissIndexHandle index, FaissIndexHandle other, long addId);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVF_copy_subset_to(FaissIndexHandle index, FaissIndexHandle other, int subsetType, long a1, long a2);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVF_search_preassigned(FaissIndexHandle index, long n, IntPtr x, long k, IntPtr assign, IntPtr centroidDis, IntPtr distances, IntPtr labels, [MarshalAs(UnmanagedType.Bool)] bool storePairs);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexIVF_get_list_size(FaissIndexHandle index, UIntPtr listNo);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVF_make_direct_map(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool maintainDirectMap);
    
    [LibraryImport(LibraryName)]
    internal static partial double faiss_IndexIVF_imbalance_factor(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVF_print_stats(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVF_invlists_get_ids(FaissIndexHandle index, UIntPtr listNo, IntPtr invlist);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVF_train_encoder(FaissIndexHandle index, long n, IntPtr x, IntPtr assign);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFScalarQuantizer_new_with(out IntPtr pIndex, FaissIndexHandle quantizer, long d, UIntPtr nlist, QuantizerType qt);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFScalarQuantizer_new_with_metric(out IntPtr pIndex, FaissIndexHandle quantizer, UIntPtr d, UIntPtr nlist, QuantizerType qt, MetricType metric, [MarshalAs(UnmanagedType.Bool)] bool encodeResidual);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexIVFScalarQuantizer_nlist(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexIVFScalarQuantizer_nprobe(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVFScalarQuantizer_set_nprobe(FaissIndexHandle index, UIntPtr nprobe);
    
    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIVFScalarQuantizer_quantizer(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFScalarQuantizer_own_fields(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVFScalarQuantizer_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFFlat_new(out IntPtr pIndex);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFFlat_new_with(out IntPtr pIndex, FaissIndexHandle quantizer, UIntPtr d, UIntPtr nlist);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFFlat_new_with_metric(out IntPtr pIndex, FaissIndexHandle quantizer, UIntPtr d, UIntPtr nlist, MetricType metric);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexIVFFlat_nlist(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_IndexIVFFlat_nprobe(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVFFlat_set_nprobe(FaissIndexHandle index, UIntPtr nprobe);
    
    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexIVFFlat_quantizer(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial sbyte faiss_IndexIVFFlat_quantizer_trains_alone(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFFlat_own_fields(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexIVFFlat_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexIVFFlat_update_vectors(FaissIndexHandle index, int nv, long* idx, float* v);
}