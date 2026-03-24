namespace Faiss.Gpu.Interfaces;

using Faiss.Cpu.Interfaces;


public interface INativeFaissGpuIndex<T> : INativeFaissIndex<T> where T : INativeFaissIndex
{
    
}