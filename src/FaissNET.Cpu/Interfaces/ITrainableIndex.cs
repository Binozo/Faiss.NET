namespace Faiss.Cpu.Interfaces;

public interface ITrainableIndex
{
    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    public bool IsTrained { get; }
}