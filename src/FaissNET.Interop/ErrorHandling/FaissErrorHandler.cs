namespace Faiss.Interop.ErrorHandling;

using System.Runtime.InteropServices;

using Exceptions;
using NativeMethods;

internal static class FaissErrorHandler
{
    /// <summary>
    /// Validates the result code from a native Faiss call.
    /// </summary>
    /// <exception cref="FaissException">Thrown if the code indicates a failure.</exception>
    public static void ThrowIfError(int resultCode)
    {
        if (resultCode == 0)
        {
            return;
        }

        IntPtr errorPtr = Native.faiss_get_last_error();
        string errorMessage = Marshal.PtrToStringAnsi(errorPtr) ?? "Unknown Faiss error occurred.";
        
        throw new FaissException($"Faiss native error (Code {resultCode}): {errorMessage}");
    }
}