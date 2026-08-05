namespace Faiss.Cpu.Interfaces;

public interface IOnGpuIndex<T> where T : INativeIndex<T>, IFromNativeHandle<T>
{
    
}