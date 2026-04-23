using System.Runtime.InteropServices;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Models;

namespace Faiss.Cpu.Serializer;

public static class BinaryDeserializer
{
    public static T Read<T>(string filePath, IoFlags flags = IoFlags.None) where T : INativeBinaryIndex<T>, IFromNativeBinaryHandle<T>
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_read_index_binary_fname(filePath, (int)flags, out IntPtr indexPtr)
        );
        return T.FromHandle(indexPtr);
    }

    public static T Read<T>(Stream stream, IoFlags flags = IoFlags.None) where T : INativeBinaryIndex<T>, IFromNativeBinaryHandle<T>
    {
        Native.CustomIoReaderCallback readerCallback = (ptr, size, nitems) =>
        {
            long totalBytes = (long)(size * nitems);
            if (totalBytes == 0) return 0;
            unsafe
            {
                var span = new Span<byte>(ptr.ToPointer(), (int)totalBytes);
                int bytesRead = stream.Read(span);
                return (nuint)(bytesRead / (long)size);
            }
        };
        
        GCHandle handle = GCHandle.Alloc(readerCallback);
        try
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_CustomIOReader_new(out IntPtr ioReader, readerCallback)
            );
            try
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_read_index_binary_custom(ioReader, (int)flags, out IntPtr indexPtr)
                );
                return T.FromHandle(indexPtr);
            }
            finally
            {
                Native.faiss_CustomIOReader_free(ioReader);
            }
        }
        finally
        {
            handle.Free();
        }
    }
}