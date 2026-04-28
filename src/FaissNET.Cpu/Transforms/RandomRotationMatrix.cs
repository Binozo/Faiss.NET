using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

public sealed class RandomRotationMatrix : VectorTransform
{
    public RandomRotationMatrix(int dIn, int dOut) : base(CreateHandle(dIn, dOut))
    {
    }

    private static IntPtr CreateHandle(int dIn, int dOut)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_RandomRotationMatrix_new_with(out IntPtr ptr, dIn, dOut));

        return ptr;
    }
}