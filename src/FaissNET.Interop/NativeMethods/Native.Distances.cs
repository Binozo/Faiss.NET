using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_pairwise_L2sqr(long d, long nq, ReadOnlySpan<float> xq, long nb, ReadOnlySpan<float> xb, Span<float> dis, long ldq, long ldb, long ldd);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial float faiss_fvec_norm_L2sqr(ReadOnlySpan<float> x, nuint d);
    
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_fvec_norms_L2(Span<float> norms, ReadOnlySpan<float> x, nuint d, nuint nx);
    
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_fvec_norms_L2sqr(Span<float> norms, ReadOnlySpan<float> x, nuint d, nuint nx);
    
    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_fvec_renorm_L2(nuint d, nuint nx, Span<float> x);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial int faiss_get_distance_compute_blas_threshold();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_set_distance_compute_blas_threshold(int value);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial int faiss_get_distance_compute_blas_query_bs();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_set_distance_compute_blas_query_bs(int value);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial int faiss_get_distance_compute_blas_database_bs();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_set_distance_compute_blas_database_bs(int value);

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial int faiss_get_distance_compute_min_k_reservoir();

    [LibraryImport(LibraryName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    public static partial void faiss_set_distance_compute_min_k_reservoir(int value);
}