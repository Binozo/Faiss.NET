namespace Faiss.Cpu.Interfaces;

public interface IMultiGpuClonableIndex<TCpu, out TGpu> : INativeIndex where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
{
    internal virtual bool IsMultiGpuClonable() => true;
}