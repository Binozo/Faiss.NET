namespace Faiss.Exceptions;

public class FaissException : Exception
{
    public FaissException(string message) : base(message)
    {
    }

    public FaissException(string message, Exception innerException) : base(message, innerException)
    {
    }
}