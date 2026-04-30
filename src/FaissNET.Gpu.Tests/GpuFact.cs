using Xunit;

namespace Faiss.Gpu.Tests;

public static class GpuFact
{
    public static void SkipIfNoGpu()
    {
        try
        {
            if (FaissGpu.GetNumGpus() == 0)
            {
                Assert.Skip("No GPU available.");
            }
        }
        catch (Exception ex)
        {
            Assert.Skip($"Unable to query GPU availability: {ex.Message}");
        }
    }

    public static void SkipIfDeviceUnavailable(int deviceId)
    {
        try
        {
            int count = FaissGpu.GetNumGpus();
            if (count == 0)
            {
                Assert.Skip("No GPU available.");
            }

            if (deviceId >= count)
            {
                Assert.Skip($"GPU device {deviceId} is not available (found {count} GPU(s)).");
            }
        }
        catch (Exception ex)
        {
            Assert.Skip($"Unable to query GPU availability: {ex.Message}");
        }
    }
}
