using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IClonableFloatIndex<T> : INativeIndex, IFloatIndex where T : IFloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    /// <summary>
    /// Creates a copy of the index.
    /// </summary>
    public T Clone()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_clone_index(Handle, out IntPtr ptr));
        return T.FromPointer(ptr);
    }
}