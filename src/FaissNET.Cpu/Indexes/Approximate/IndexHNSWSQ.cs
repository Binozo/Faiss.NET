using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// HNSW + Scalar Quantization. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexHNSWSQ : CpuIndex<IndexHNSWSQ>, IFromNativeHandle<IndexHNSWSQ>
{
    private IndexHNSWSQ(IntPtr handle) : base(handle) { }
    static IndexHNSWSQ IFromNativeHandle<IndexHNSWSQ>.FromHandle(IntPtr handle) => new(handle);
}