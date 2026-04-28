using System.Runtime.CompilerServices;

namespace Faiss.Interop.Errors;

using System.Runtime.InteropServices;

using Exceptions;
using NativeMethods;

internal static class FaissErrorHandler
{
    /// <summary>
    /// Validates the result code from a native Faiss call.
    /// </summary>
    /// <exception cref="FaissNativeException">Thrown for Faiss library exceptions (code -2).</exception>
    /// <exception cref="FaissRuntimeException">Thrown for standard C++ exceptions (code -4).</exception>
    /// <exception cref="FaissUnknownException">Thrown for unknown exceptions (code -1).</exception>
    public static void ThrowIfError(int resultCode, [CallerMemberName] string operation = "")
    {
        if (resultCode == 0)
        {
            return;
        }
        IntPtr errorPtr = Native.faiss_get_last_error();
        string errorMessage = Marshal.PtrToStringAnsi(errorPtr) ?? "Unknown Faiss error occurred.";
        string fullMessage = string.IsNullOrEmpty(operation)
            ? errorMessage
            : $"{operation} failed: {errorMessage}";

        throw resultCode switch
        {
            -2 => new FaissNativeException(fullMessage),
            -4 => new FaissRuntimeException(fullMessage),
            _ => new FaissUnknownException($"{fullMessage} (code {resultCode})")
        };
    }
}