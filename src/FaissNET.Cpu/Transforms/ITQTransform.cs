using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Complete ITQ pipeline in one transform: mean-centering + L2 normalization, optional PCA to d_out (do_pca), then an ITQ rotation.
/// Merged into a single linear map at train time. Not reversible.
/// </summary>
public sealed class ITQTransform : VectorTransform
{
    public ITQTransform(int dIn, int dOut, bool doPca = true)
        : base(CreateHandle(dIn, dOut, doPca))
    {
    }

    private static IntPtr CreateHandle(int dIn, int dOut, bool doPca)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_ITQTransform_new_with(out IntPtr ptr, dIn, dOut, doPca));

        return ptr;
    }

    public bool DoPca => Native.faiss_ITQTransform_do_pca(Handle) != 0;
}