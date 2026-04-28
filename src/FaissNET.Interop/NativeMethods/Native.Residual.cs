using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_compute_residual(FaissIndexHandle index, float* x, float* residual, long key);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_compute_residual_n(FaissIndexHandle index, long n, float* x, float* residuals, long* keys);
}