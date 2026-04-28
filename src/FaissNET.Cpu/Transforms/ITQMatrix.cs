using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

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