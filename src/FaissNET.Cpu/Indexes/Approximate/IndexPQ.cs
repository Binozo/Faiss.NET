using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// Product Quantization flat index. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexPQ : CpuIndex<IndexPQ>, IFromNativeHandle<IndexPQ>
{
    private IndexPQ(IntPtr handle) : base(handle) { }
    static IndexPQ IFromNativeHandle<IndexPQ>.FromHandle(IntPtr handle) => new(handle);
}