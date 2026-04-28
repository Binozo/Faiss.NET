using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

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

    public bool DoPca => Native.faiss_ITQTransform_do_pca(SafeHandle.DangerousGetHandle()) != 0;
}