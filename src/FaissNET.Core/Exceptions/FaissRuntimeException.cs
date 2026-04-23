namespace Faiss.Exceptions;

public class FaissRuntimeException : FaissException
{
    public FaissRuntimeException(string message) : base(message)
    {
    }

    public FaissRuntimeException(string message, Exception innerException) : base(message, innerException)
    {
    }
}