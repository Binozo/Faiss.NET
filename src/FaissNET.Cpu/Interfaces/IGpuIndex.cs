namespace Faiss.Cpu.Interfaces;

public interface IGpuIndex<T> : INativeIndex where T : INativeIndex, IFromNativeIndexHandle<T>
{
    
}