using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Indexes.Binary;

public abstract class CpuBinaryIndex<T> : BinaryIndex<T>, ICpuBinaryIndex<T> where T : BinaryIndex<T>, INativeBinaryIndex<T>, IFromNativeBinaryHandle<T>
{
    protected CpuBinaryIndex()
    {
        
    }

    protected CpuBinaryIndex(IntPtr handle) : base(handle)
    {
        
    }

    public T Clone()
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_clone_index_binary(SafeHandle, out IntPtr clonedPtr)
        );

        return T.FromHandle(clonedPtr);
    }
}