using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Iterative Quantization rotation (Gong et al., PAMI’12): learns an orthonormal rotation that minimizes the quantization error
/// of sign-binarizing the output, i.e. produces vectors meant to be turned into binary codes.
/// Square (d → d), max_iter SVD refinement rounds.
/// <see cref="VectorTransform.ReverseTransform"/> is unavailable (the orthonormality flag is never set).
/// </summary>
public sealed class ITQMatrix : VectorTransform
{
    public ITQMatrix(int d) : base(CreateHandle(d))
    {
    }

    private static IntPtr CreateHandle(int d)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_ITQMatrix_new_with(out IntPtr ptr, d));

        return ptr;
    }
}