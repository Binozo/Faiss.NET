using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IClonableBinaryIndex<T> : INativeBinaryIndex, IBinaryIndex where T : IBinaryIndex, INativeBinaryIndex, IFromNativeBinaryIndexHandle<T>
{
    /// <summary>
    /// Creates a copy of the index.
    /// </summary>
    public T Clone();
}

internal static class ClonableBinaryIndexImpl<T> where T : IBinaryIndex, INativeBinaryIndex, IFromNativeBinaryIndexHandle<T>
{
    public static T Clone(INativeBinaryIndex index)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_clone_index_binary(index.Handle, out IntPtr ptr));
        return T.FromPointer(ptr);
    }
}