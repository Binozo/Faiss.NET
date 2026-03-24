namespace Faiss.Interop.NativeMethods;

using System;
using System.Runtime.InteropServices;

internal static partial class FaissIONativeMethods
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_write_index_fname(IntPtr idx, string fname);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_read_index_fname(string fname, int io_flags, out IntPtr p_out);
    
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate nuint CustomIOWriterCallback(IntPtr ptr, nuint size, nuint nitems);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate nuint CustomIOReaderCallback(IntPtr ptr, nuint size, nuint nitems);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_CustomIOWriter_new(out IntPtr p_out, CustomIOWriterCallback func_in);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_CustomIOReader_new(out IntPtr p_out, CustomIOReaderCallback func_in);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_write_index_custom(IntPtr idx, IntPtr io_writer, int io_flags);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_read_index_custom(IntPtr io_reader, int io_flags, out IntPtr p_out);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_CustomIOWriter_free(IntPtr obj);
    
    [LibraryImport(LibraryName)]
    internal static partial void faiss_CustomIOReader_free(IntPtr obj);
}