using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Zero-cost dimension remap: output component j copies input component map[j], or 0 where map[j] == −1. Used to truncate, zero-pad (Pad factory prefix) or reorder dimensions without any arithmetic.
/// Needs no training.
/// <see cref="VectorTransform.ReverseTransform"/> is a scatter-back and is exact only when the map is a permutation.
/// </summary>
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