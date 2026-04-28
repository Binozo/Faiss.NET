using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryHash : CpuBinaryIndex<IndexBinaryHash>, IFromNativeBinaryHandle<IndexBinaryHash>
{
    public IndexBinaryHash(int dimensions, int b)
    {
        string description = $"BHash{b}";
        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, description)
        );

        SafeHandle = new FaissIndexBinaryHandle(ptr);
    }

    private IndexBinaryHash(IntPtr handle) : base(handle)
    {
    }

    static IndexBinaryHash IFromNativeBinaryHandle<IndexBinaryHash>.FromHandle(IntPtr handle) => new(handle);
}