using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.IVF;

internal readonly struct IndexIVFRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexIVF_free(handle);
}