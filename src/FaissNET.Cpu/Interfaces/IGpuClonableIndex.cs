namespace Faiss.Cpu.Interfaces;

public interface IGpuClonableIndex<TCpu, out TGpu> : INativeIndex where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu> where TGpu : INativeIndex, IFromNativeIndexHandle<TGpu>//, IGpuIndex<TCpu>
{
    internal bool IsGpuClonable() => true;

    /// <summary>Hands the wrapper of the fresh device copy the source facts it cannot read back from a GPU index.</summary>
    internal void OnClonedToGpu(INativeIndex gpuIndex)
    {
    }
}