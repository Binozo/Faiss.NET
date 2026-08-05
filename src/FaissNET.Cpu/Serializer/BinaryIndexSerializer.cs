using System.Runtime.InteropServices;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Serializer;

public static class BinaryIndexSerializer
{
    public static void Write(INativeBinaryIndex index, string filePath)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_write_index_binary_fname(index.Handle, filePath)
        );
    }

    public static void Write(INativeBinaryIndex index, Stream stream)
    {
        Native.CustomIoWriterCallback writerCallback = (ptr, size, nitems) =>
        {
            long totalBytes = (long)(size * nitems);
            if (totalBytes == 0) return 0;
            unsafe
            {
                var span = new ReadOnlySpan<byte>(ptr.ToPointer(), (int)totalBytes);
                stream.Write(span);
            }

            return nitems;
        };

        GCHandle handle = GCHandle.Alloc(writerCallback);
        try
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_CustomIOWriter_new(out IntPtr ioWriter, writerCallback)
            );
            try
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_write_index_binary_custom(index.Handle, ioWriter)
                );
            }
            finally
            {
                Native.faiss_CustomIOWriter_free(ioWriter);
            }
        }
        finally
        {
            handle.Free();
        }
    }
}