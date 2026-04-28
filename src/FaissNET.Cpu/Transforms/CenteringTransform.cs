using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

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