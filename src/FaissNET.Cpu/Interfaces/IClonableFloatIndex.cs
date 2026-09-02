using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IClonableFloatIndex<T> : INativeIndex, IFloatIndex where T : IFloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    /// <summary>
    /// Creates a copy of the index.
    /// </summary>
    public T Clone();
}

internal static class ClonableFloatIndexImpl<T> where T : IFloatIndex, INativeIndex, IFromNativeIndexHandle<T>
{
    public static T Clone(INativeIndex index)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_clone_index(index.Handle, out IntPtr ptr));
        return T.FromPointer(ptr);
    }
}