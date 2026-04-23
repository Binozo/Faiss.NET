namespace Faiss.Interfaces;

public interface ISearchParameters : IDisposable { }

internal interface INativeSearchParameters : ISearchParameters
{
    IntPtr DangerousGetHandle();
}