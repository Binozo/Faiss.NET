namespace Faiss.Interfaces;

public interface INativeIndex : IFaissIndex
{
    IntPtr Handle { get; }
}