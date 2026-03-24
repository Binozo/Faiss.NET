namespace Faiss.Cpu.Interfaces;

public interface INativeFaissCpuIndex : INativeFaissIndex
{
    static abstract INativeFaissIndex FromHandle(IntPtr handle);
}