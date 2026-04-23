using Faiss.Exceptions;

namespace Faiss.Cpu.Exceptions;

/// <summary>
/// Thrown when a user tries to add vectors to an index that requires training (like IVF)
/// before actually calling TrainAsync().
/// </summary>
public class FaissUntrainedException : FaissException
{
    public FaissUntrainedException(string message = "Index must be trained before adding vectors.")
        : base(message)
    {
    }
}
