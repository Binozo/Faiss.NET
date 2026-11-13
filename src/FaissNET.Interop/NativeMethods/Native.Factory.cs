using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static partial class Native
{
    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_index_factory(out IntPtr pIndex, int d, string description, MetricType metric);
    
    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_index_binary_factory(out IntPtr pIndex, int d, string description);
    
    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_reverse_index_factory(FaissIndexHandle index, byte* buf, UIntPtr bufSize);
    
    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_reverse_index_binary_factory(FaissBinaryIndexHandle index, byte* buf, UIntPtr bufSize);
}