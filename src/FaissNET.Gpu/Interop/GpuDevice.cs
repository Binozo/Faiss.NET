using System.Runtime.CompilerServices;

namespace Faiss.Gpu.Interop;

internal static class GpuDevice
{
    /// <summary>
    /// Rejects a device id that names no visible GPU.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="device"/> is negative or not below <see cref="FaissGpu.GetNumGpus"/>.</exception>
    public static void ThrowIfInvalid(int device, [CallerArgumentExpression(nameof(device))] string? paramName = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(device, paramName);

        var count = FaissGpu.GetNumGpus();
        if (device >= count)
            throw new ArgumentOutOfRangeException(paramName, device, $"Device {device} does not exist; {count} GPU(s) are visible.");
    }
}
