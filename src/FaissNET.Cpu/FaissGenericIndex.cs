namespace Faiss.Cpu;

using Models;
using Interfaces;
using Indexes.Descriptor;
using Interop.SafeHandles;
using Interop.ErrorHandling;
using Interop.NativeMethods;

public sealed class FaissGenericIndex : FaissCpuIndex, INativeFaissCpuIndex
{
    private protected override FaissIndexHandle NativeHandle { get; }

    internal FaissGenericIndex(IntPtr ptr)
    {
        NativeHandle = new FaissIndexHandle(ptr);
    }

    public FaissGenericIndex(int dimensions, FaissDescriptorBuilder descriptionBuilder, MetricType metric)
    {
        FaissErrorHandler.ThrowIfError(IndexFactoryNativeMethods.faiss_index_factory(out IntPtr ptr, dimensions, descriptionBuilder.Build(), metric));
        NativeHandle = new FaissIndexHandle(ptr);
    }

    public static INativeFaissIndex FromHandle(IntPtr handle)
    {
        return new FaissGenericIndex(handle);
    }
}