using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// IVF + Product Quantization index. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexIVFPQ : CpuIndex<IndexIVFPQ>, IFromNativeHandle<IndexIVFPQ>
{
    private IndexIVFPQ(IntPtr handle) : base(handle) { }
    static IndexIVFPQ IFromNativeHandle<IndexIVFPQ>.FromHandle(IntPtr handle) => new(handle);
}