namespace Faiss.Cpu.Indexes;

using Descriptor;
using Models;

using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;

/// <summary>
/// Hierarchical Navigable Small World index. 
/// The industry standard for fast approximate nearest neighbor search.
/// </summary>
public sealed class FaissIndexHNSW : FaissCpuIndex
{
    private protected override FaissIndexHandle NativeHandle { get; }

    public FaissIndexHNSW(int dimensions, int m = 32, MetricType metricType = MetricType.L2)
    {
        var description = FaissDescriptorBuilder.Create()
            .WithHNSW(m)
            .EncodeWithFlat();

        FaissErrorHandler.ThrowIfError(
            IndexFactoryNativeMethods.faiss_index_factory(out IntPtr ptr, dimensions, description.Build(), metricType)
        );

        NativeHandle = new FaissIndexHandle(ptr);
    }
}