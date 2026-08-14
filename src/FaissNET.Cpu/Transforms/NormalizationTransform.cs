using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Normalizes each vector to unit L2 norm (only norm = 2.0 is implemented).
/// Needs no training.
/// <see cref="VectorTransform.ReverseTransform"/> is the identity. Original norms are not recoverable.
/// </summary>
public sealed class NormalizationTransform : VectorTransform
{
    public NormalizationTransform(int dimension, float norm = 2.0f) : base(CreateHandle(dimension, norm))
    {
    }

    private static IntPtr CreateHandle(int d, float norm)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_NormalizationTransform_new_with(out IntPtr ptr, d, norm));

        return ptr;
    }

    public float Norm => Native.faiss_NormalizationTransform_norm(Handle);
}