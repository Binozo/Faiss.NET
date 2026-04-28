using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// HNSW + Product Quantization. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexHNSWPQ : CpuIndex<IndexHNSWPQ>, IFromNativeHandle<IndexHNSWPQ>
{
    private IndexHNSWPQ(IntPtr handle) : base(handle) { }
    static IndexHNSWPQ IFromNativeHandle<IndexHNSWPQ>.FromHandle(IntPtr handle) => new(handle);
}