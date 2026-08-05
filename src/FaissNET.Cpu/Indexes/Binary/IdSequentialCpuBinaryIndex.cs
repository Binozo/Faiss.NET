using Faiss.Cpu.Exceptions;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Indexes.Binary;

/// <inheritdoc cref="CpuBinaryIndex{T}" />
public class IdSequentialCpuBinaryIndex<T> : CpuBinaryIndex<T> where T : CpuBinaryIndex<T>, INativeBinaryIndex<T>, IFromNativeBinaryHandle<T>, IIdSequentialBinaryIndex
{
    protected IdSequentialCpuBinaryIndex()
    {
        
    }

    protected IdSequentialCpuBinaryIndex(IntPtr handle) : base(handle)
    {
        
    }

    public unsafe void Add(long count, ReadOnlySpan<byte> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        fixed (byte* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinary_add(SafeHandle, count, pVectors));
        }
    }
}