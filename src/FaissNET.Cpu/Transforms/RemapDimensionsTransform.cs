using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

public sealed class RemapDimensionsTransform : VectorTransform
{
    public RemapDimensionsTransform(int dIn, int dOut, bool uniform = false)
        : base(CreateHandle(dIn, dOut, uniform))
    {
    }

    private static IntPtr CreateHandle(int dIn, int dOut, bool uniform)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_RemapDimensionsTransform_new_with(out IntPtr ptr, dIn, dOut, uniform)
        );

        return ptr;
    }
}