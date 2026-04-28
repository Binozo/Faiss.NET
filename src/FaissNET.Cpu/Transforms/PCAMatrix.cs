using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

public sealed class PCAMatrix : VectorTransform
{
    public PCAMatrix(int dIn, int dOut, float eigenPower = 0, bool randomRotation = false)
        : base(CreateHandle(dIn, dOut, eigenPower, randomRotation))
    {
    }

    private static IntPtr CreateHandle(int dIn, int dOut, float eigenPower, bool randomRotation)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_PCAMatrix_new_with(out IntPtr ptr, dIn, dOut, eigenPower, randomRotation)
        );

        return ptr;
    }

    public float EigenPower => Native.faiss_PCAMatrix_eigen_power(SafeHandle.DangerousGetHandle());
    public bool RandomRotation => Native.faiss_PCAMatrix_random_rotation(SafeHandle.DangerousGetHandle()) != 0;

    public int BalancedBins
    {
        get => Native.faiss_PCAMatrix_balanced_bins(SafeHandle.DangerousGetHandle());
        set => Native.faiss_PCAMatrix_set_balanced_bins(SafeHandle.DangerousGetHandle(), value);
    }
}