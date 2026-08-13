using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial void faiss_VectorTransform_free(IntPtr vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_VectorTransform_train(FaissVectorTransformHandle vt, long n, float* x);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_VectorTransform_apply(FaissVectorTransformHandle vt, long n, float* x);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_VectorTransform_apply_noalloc(FaissVectorTransformHandle vt, long n, float* x, float* xt);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_VectorTransform_reverse_transform(FaissVectorTransformHandle vt, long n, float* xt, float* x);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_VectorTransform_is_trained(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_VectorTransform_d_in(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_VectorTransform_d_out(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_RandomRotationMatrix_new_with(out IntPtr pVt, int dIn, int dOut);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_PCAMatrix_new_with(out IntPtr pVt, int dIn, int dOut, float eigenPower, [MarshalAs(UnmanagedType.Bool)] bool randomRotation);

    [LibraryImport(LibraryName)]
    internal static partial float faiss_PCAMatrix_eigen_power(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_PCAMatrix_random_rotation(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_PCAMatrix_balanced_bins(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_PCAMatrix_set_balanced_bins(FaissVectorTransformHandle vt, int balancedBins);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ITQMatrix_new_with(out IntPtr pVt, int d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ITQTransform_new_with(out IntPtr pVt, int dIn, int dOut, [MarshalAs(UnmanagedType.Bool)] bool doPca);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ITQTransform_do_pca(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_OPQMatrix_new_with(out IntPtr pVt, int d, int m, int d2);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_OPQMatrix_verbose(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_OPQMatrix_set_verbose(FaissVectorTransformHandle vt, [MarshalAs(UnmanagedType.Bool)] bool verbose);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_OPQMatrix_niter(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_OPQMatrix_set_niter(FaissVectorTransformHandle vt, int niter);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_OPQMatrix_niter_pq(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_OPQMatrix_set_niter_pq(FaissVectorTransformHandle vt, int niterPq);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_RemapDimensionsTransform_new_with(out IntPtr pVt, int dIn, int dOut, [MarshalAs(UnmanagedType.Bool)] bool uniform);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_NormalizationTransform_new_with(out IntPtr pVt, int d, float norm);

    [LibraryImport(LibraryName)]
    internal static partial float faiss_NormalizationTransform_norm(FaissVectorTransformHandle vt);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_CenteringTransform_new_with(out IntPtr pVt, int d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexPreTransform_new_with_transform(out IntPtr pIndex, FaissVectorTransformHandle ltrans, FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexPreTransform_new_with(out IntPtr pIndex, FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexPreTransform_prepend_transform(FaissIndexHandle index, FaissVectorTransformHandle ltrans);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_IndexPreTransform_index(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexPreTransform_own_fields(FaissIndexHandle index);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexPreTransform_set_own_fields(FaissIndexHandle index, [MarshalAs(UnmanagedType.Bool)] bool ownFields);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_IndexPreTransform_free(IntPtr index);
}