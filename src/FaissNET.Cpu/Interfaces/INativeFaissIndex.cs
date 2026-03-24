namespace Faiss.Cpu.Interfaces;

using Faiss.Interfaces;

public interface INativeFaissIndex : IFaissIndex
{
    IntPtr Handle { get; }
}

public interface INativeFaissIndex<T> : INativeFaissIndex where T : INativeFaissIndex
{
}