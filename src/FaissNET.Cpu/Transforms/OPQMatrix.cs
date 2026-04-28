using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

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

    public bool Verbose
    {
        get => Native.faiss_OPQMatrix_verbose(SafeHandle.DangerousGetHandle()) != 0;
        set => Native.faiss_OPQMatrix_set_verbose(SafeHandle.DangerousGetHandle(), value);
    }

    public int Niter
    {
        get => Native.faiss_OPQMatrix_niter(SafeHandle.DangerousGetHandle());
        set => Native.faiss_OPQMatrix_set_niter(SafeHandle.DangerousGetHandle(), value);
    }

    public int NiterPq
    {
        get => Native.faiss_OPQMatrix_niter_pq(SafeHandle.DangerousGetHandle());
        set => Native.faiss_OPQMatrix_set_niter_pq(SafeHandle.DangerousGetHandle(), value);
    }
}