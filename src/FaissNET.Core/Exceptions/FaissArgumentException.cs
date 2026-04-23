namespace Faiss.Exceptions;

public class FaissArgumentException : FaissException
{
    public FaissArgumentException(string message) : base(message)
    {
    }

    public FaissArgumentException(string message, Exception innerException) : base(message, innerException)
    {
    }
}