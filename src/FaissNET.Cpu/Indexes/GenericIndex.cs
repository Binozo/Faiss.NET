using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Indexes;

public sealed class GenericIndex : CpuIndex<GenericIndex>, IFromNativeHandle<GenericIndex>
{
    internal GenericIndex(IntPtr handle) : base(handle)
    {

    }

    static GenericIndex IFromNativeHandle<GenericIndex>.FromHandle(IntPtr handle) => new(handle);
}
