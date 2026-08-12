namespace Faiss.Cpu.Interfaces;

public interface IGpuClonableIndex<TCpu, out TGpu> : INativeIndex where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
{
    internal virtual bool IsGpuClonable() => true;
}