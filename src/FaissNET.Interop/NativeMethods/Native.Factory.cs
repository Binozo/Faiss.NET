using System.Runtime.InteropServices;
using Faiss.Models;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_index_factory(out IntPtr pIndex, int d, string description, MetricType metric);
    
    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_index_binary_factory(out IntPtr pIndex, int d, string description);
}