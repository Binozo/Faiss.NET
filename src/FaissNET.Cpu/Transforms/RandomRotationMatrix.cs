using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Random orthonormal rotation (QR of a Gaussian matrix; a tight frame when d_out > d_in).
/// Training is data-independent (<see cref="VectorTransform.Train"/> just seeds the generator deterministically (12345)).
/// Preserves L2 norms and distances, so nearest neighbors are unchanged.
/// Used to spread information across dimensions before quantization.
/// Exactly reversible via transpose.
/// </summary>
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