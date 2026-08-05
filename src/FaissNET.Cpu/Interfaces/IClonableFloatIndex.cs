using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Interfaces;

public interface IClonableIndex<out T> : INativeIndex, IFloatIndex where T : IFloatIndex, INativeIndex<T>, IFromNativeHandle<T>
{
    /// <summary>
    /// Creates a copy of the index.
    /// </summary>
    public T Clone()
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_clone_index(Handle, out FaissIndexHandle clonedPtr)
        );

        return T.FromHandle(clonedPtr);
    }
}