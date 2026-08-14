using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Optimized Product Quantization rotation (Ge et al., CVPR’13): learns an orthonormal rotation that minimizes the reconstruction error of a Product Quantizer with M subquantizers applied after the transform.
/// Place before IndexPQ/IndexIVFPQ with matching M. Training alternates PQ fitting and SVD updates (niter rounds) and is the most expensive transform to train.
/// Reversible (orthonormal) once trained.
/// </summary>
public sealed class OPQMatrix : VectorTransform
{
    public OPQMatrix(int d, int m, int d2) : base(CreateHandle(d, m, d2))
    {
    }

    private static IntPtr CreateHandle(int d, int m, int d2)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_OPQMatrix_new_with(out IntPtr ptr, d, m, d2));

        return ptr;
    }

    public int Niter
    {
        get => Native.faiss_OPQMatrix_niter(Handle);
        set => Native.faiss_OPQMatrix_set_niter(Handle, value);
    }

    public int NiterPq
    {
        get => Native.faiss_OPQMatrix_niter_pq(Handle);
        set => Native.faiss_OPQMatrix_set_niter_pq(Handle, value);
    }

    public bool Verbose
    {
        get => Native.faiss_OPQMatrix_verbose(Handle) != 0;
        set => Native.faiss_OPQMatrix_set_verbose(Handle, value);
    }
}