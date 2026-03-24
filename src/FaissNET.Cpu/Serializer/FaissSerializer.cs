namespace Faiss.Cpu.Serializer;

using System.Runtime.InteropServices;

using Interfaces;
using Interop.ErrorHandling;
using Interop.NativeMethods;

/// <summary>
/// Faiss index serializer.
/// </summary>
public static class FaissSerializer
{
    public static void Write(INativeFaissIndex index, string filePath)
    {
        FaissErrorHandler.ThrowIfError(
            FaissIONativeMethods.faiss_write_index_fname(index.Handle, filePath)
        );
    }

    public static T Read<T>(string filePath) where T : INativeFaissCpuIndex
    {
        FaissErrorHandler.ThrowIfError(
            FaissIONativeMethods.faiss_read_index_fname(filePath, 0, out IntPtr indexPtr)
        );
        
        return (T)T.FromHandle(indexPtr);
    }

    public static void Write(INativeFaissIndex index, Stream stream)
    {
        FaissIONativeMethods.CustomIOWriterCallback writerCallback = (ptr, size, nitems) =>
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
                FaissIONativeMethods.faiss_CustomIOWriter_new(out IntPtr ioWriter, writerCallback)
            );

            try
            {
                FaissErrorHandler.ThrowIfError(
                    FaissIONativeMethods.faiss_write_index_custom(index.Handle, ioWriter, 0)
                );
            }
            finally
            {
                FaissIONativeMethods.faiss_CustomIOWriter_free(ioWriter);
            }
        }
        finally
        {
            handle.Free();
        }
    }
    
    public static T Read<T>(Stream stream) where T : INativeFaissCpuIndex
    {
        FaissIONativeMethods.CustomIOReaderCallback readerCallback = (ptr, size, nitems) =>
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
                FaissIONativeMethods.faiss_CustomIOReader_new(out IntPtr ioReader, readerCallback)
            );

            try
            {
                FaissErrorHandler.ThrowIfError(
                    FaissIONativeMethods.faiss_read_index_custom(ioReader, 0, out IntPtr indexPtr)
                );

                return (T)T.FromHandle(indexPtr);
            }
            finally
            {
                FaissIONativeMethods.faiss_CustomIOReader_free(ioReader);
            }
        }
        finally
        {
            handle.Free();
        }
    }
}