using Faiss.Gpu.Resources;

namespace Faiss.Gpu.Interfaces;

/// <summary>
/// Abstracts a provider of native GPU resources for Faiss.
/// </summary>
public interface IGpuResourcesProvider
{
    /// <summary>
    /// Gets the native handle for this provider.
    /// </summary>
    /// <remarks>
    /// This is the raw pointer passed to native transfer functions
    /// such as <c>faiss_index_cpu_to_gpu</c>.
    /// </remarks>
    IntPtr NativeHandle { get; }

    /// <summary>
    /// Obtains a scoped reference to the underlying native GPU resources.
    /// </summary>
    /// <returns>
    /// A <see cref="GpuResources"/> that is valid only for the current
    /// stack frame and guaranteed not to outlive this provider.
    /// </returns>
    GpuResources GetResources();
}