using Faiss.Cpu.Interfaces;

namespace Faiss.Gpu.Interfaces;

public interface INativeGpuIndex<T> : INativeIndex where T : INativeIndex, IFromNativeIndexHandle<T>
{
}