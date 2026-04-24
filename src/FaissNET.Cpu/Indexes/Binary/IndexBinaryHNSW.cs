using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHNSW : BinaryIndex<IndexBinaryHNSW>, IFromNativeBinaryHandle<IndexBinaryHNSW>
{
    public IndexBinaryHNSW(int dimensions, int m = 32)
    {
        string description = $"BHNSW{m}";
        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, description)
        );

        SafeHandle = new FaissIndexBinaryHandle(ptr);
    }

    private IndexBinaryHNSW(IntPtr handle) : base(handle)
    {
    }

    static IndexBinaryHNSW IFromNativeBinaryHandle<IndexBinaryHNSW>.FromHandle(IntPtr handle) => new(handle);
}