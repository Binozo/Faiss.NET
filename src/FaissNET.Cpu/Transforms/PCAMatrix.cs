using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Principal Component Analysis projection onto the top-d_out eigenvectors of the data covariance (eigenvalues sorted decreasing),
/// with mean subtraction built in. <see cref="EigenPower"/> scales components by eigenvalue^power: 0 = plain PCA (orthonormal, reversible), −0.5 = full whitening (not reversible).
/// Optional random rotation after PCA (<see cref="RandomRotation"/>) and variance spreading across <see cref="BalancedBins"/> groups.
/// Output dims beyond the data rank are zero-padded when n &lt; d_in.
/// </summary>
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

    public float EigenPower => Native.faiss_PCAMatrix_eigen_power(Handle);

    public bool RandomRotation => Native.faiss_PCAMatrix_random_rotation(Handle) != 0;

    public int BalancedBins
    {
        get => Native.faiss_PCAMatrix_balanced_bins(Handle);
        set => Native.faiss_PCAMatrix_set_balanced_bins(Handle, value);
    }
}