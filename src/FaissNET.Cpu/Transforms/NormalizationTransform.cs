using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

public sealed class NormalizationTransform : VectorTransform
{
    public NormalizationTransform(int d, float norm = 1.0f) : base(CreateHandle(d, norm))
    {
    }

    private static IntPtr CreateHandle(int d, float norm)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_NormalizationTransform_new_with(out IntPtr ptr, d, norm));

        return ptr;
    }

    public float Norm => Native.faiss_NormalizationTransform_norm(SafeHandle.DangerousGetHandle());
}