using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// NSG (Navigating Spreading-out Graph) index. Created via <see cref="IndexFactory"/>.
/// </summary>
public sealed class IndexNSG : CpuIndex<IndexNSG>, IFromNativeHandle<IndexNSG>
{
    private IndexNSG(IntPtr handle) : base(handle) { }
    static IndexNSG IFromNativeHandle<IndexNSG>.FromHandle(IntPtr handle) => new(handle);
}