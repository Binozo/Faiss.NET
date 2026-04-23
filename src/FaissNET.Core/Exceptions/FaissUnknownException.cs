namespace Faiss.Exceptions;

public class FaissUnknownException : FaissException
{
    public FaissUnknownException(string message) : base(message)
    {
    }

    public FaissUnknownException(string message, Exception innerException) : base(message, innerException)
    {
    }
}