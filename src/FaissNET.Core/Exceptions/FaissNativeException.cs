namespace Faiss.Exceptions;

public class FaissNativeException : FaissException
{
    public FaissNativeException(string message) : base(message)
    {
    }

    public FaissNativeException(string message, Exception innerException) : base(message, innerException)
    {
    }
}