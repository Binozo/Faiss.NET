namespace Faiss.Exceptions;

/// <summary>
/// Thrown when a user tries to clone a cpu index to gpu while the current state of the index doesn't allow it.
/// </summary>
public class FaissGpuCloningUnsupported : FaissException
{
    public FaissGpuCloningUnsupported(string message = "Index can not be cloned to GPU.")
        : base(message)
    {
    }
}