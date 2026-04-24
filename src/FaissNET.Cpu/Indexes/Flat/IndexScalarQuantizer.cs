using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Flat;

public sealed class IndexScalarQuantizer : CpuIndex<IndexScalarQuantizer>, IFromNativeHandle<IndexScalarQuantizer>
{
    public IndexScalarQuantizer(int dimensions, QuantizerType qt, MetricType metric = MetricType.L2)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexScalarQuantizer_new_with(out IntPtr ptr, dimensions, qt, metric)
        );

        SafeHandle = new FaissIndexHandle(ptr);
    }


    private IndexScalarQuantizer(IntPtr handle) : base(handle)
    {
        
    }
    
    static IndexScalarQuantizer IFromNativeHandle<IndexScalarQuantizer>.FromHandle(IntPtr handle) => new(handle);
}