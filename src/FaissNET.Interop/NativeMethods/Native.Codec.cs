using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_Index_sa_code_size(FaissIndexHandle index, out nuint size);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_sa_encode(FaissIndexHandle index, long n, float* x, byte* bytes);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_Index_sa_decode(FaissIndexHandle index, long n, byte* bytes, float* x);
}