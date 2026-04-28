using Faiss.Cpu.Interfaces;

namespace Faiss.Gpu.Interfaces;

public interface INativeGpuIndex<T> : INativeIndex<T> where T : INativeIndex<T>, IFromNativeHandle<T>
{
}