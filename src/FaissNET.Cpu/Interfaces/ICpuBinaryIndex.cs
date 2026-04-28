using Faiss.Interfaces;

namespace Faiss.Cpu.Interfaces;

public interface ICpuBinaryIndex : IBinaryIndex
{
}

public interface ICpuBinaryIndex<T> : ICpuBinaryIndex where T : INativeBinaryIndex<T>, IFromNativeBinaryHandle<T>
{
    public T Clone();
}