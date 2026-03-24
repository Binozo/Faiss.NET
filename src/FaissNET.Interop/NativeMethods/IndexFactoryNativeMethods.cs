namespace Faiss.Interop.NativeMethods;

using System;
using System.Runtime.InteropServices;

using Models;

internal static partial class IndexFactoryNativeMethods
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_index_factory(out IntPtr p_out, int d, string description, MetricType metric);
}