using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlat1D_new(out IntPtr pIndex);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlat1D_new_with(out IntPtr pIndex, [MarshalAs(UnmanagedType.Bool)] bool continuousUpdate);
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlat1D_update_permutation(FaissIndexHandle index);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatL2_new(out FaissIndexHandle p_index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatL2_new_with(out FaissIndexHandle p_index, long d);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatIP_new(out FaissIndexHandle p_index);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexFlatIP_new_with(out FaissIndexHandle p_index, long d);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_IndexScalarQuantizer_new_with(out IntPtr pIndex, long d, QuantizerType qt, MetricType metric);
}