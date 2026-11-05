using Faiss.Interop.NativeMethods;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

/// <summary>
/// Which coarse quantizers faiss can move to a GPU together with their IVF index.
/// </summary>
/// <remarks>
/// GpuIndexIVF::copyFrom clones the quantizer with ToGpuCloner::clone_Index, which only knows IndexFlat (and subclasses)
/// and an fp16 IndexScalarQuantizer. Anything else (HNSW, PQ, other SQ types, …) throws "not implemented on GPU".
/// A single-GPU or replicated transfer reports that cleanly, but a sharded transfer clones inside an OpenMP region,
/// where the throw terminates the process.
/// </remarks>
internal static class GpuCoarseQuantizer
{
    public static bool IsSupported(FloatIndex ivf)
    {
        var handle = ivf.NativeHandle;
        var added = false;

        handle.DangerousAddRef(ref added);
        try
        {
            var quantizer = Native.faiss_IndexIVF_quantizer(handle);
            if (quantizer == IntPtr.Zero)
                return false;

            if (Native.faiss_IndexFlat_cast(quantizer) != IntPtr.Zero)
                return true;

            var sq = Native.faiss_IndexScalarQuantizer_cast(quantizer);
            return sq != IntPtr.Zero
                   && Native.faiss_ScalarQuantizer_qtype_ptr(Native.faiss_IndexScalarQuantizer_sq_ptr(sq)) == QuantizerType.QT_fp16;
        }
        finally
        {
            if (added)
                handle.DangerousRelease();
        }
    }
}
