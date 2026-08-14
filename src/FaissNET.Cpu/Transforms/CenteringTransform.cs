using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Subtracts the per-dimension data mean, learned from the training set (requires at least one training vector).
/// Used to de-bias data before other transforms or indexing. Exactly reversible by adding the mean back.
/// </summary>
public sealed class CenteringTransform : VectorTransform
{
    public CenteringTransform(int d) : base(CreateHandle(d))
    {
    }

    private static IntPtr CreateHandle(int d)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_CenteringTransform_new_with(out IntPtr ptr, d));

        return ptr;
    }
}