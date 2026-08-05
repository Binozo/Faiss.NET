namespace Faiss.Tests;

public class TestCleanupFixture : IDisposable
{
    public void Dispose()
    {
        // Trigger crashes if there is a bug in the SafeHandle handling
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}