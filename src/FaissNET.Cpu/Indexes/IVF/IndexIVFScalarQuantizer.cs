using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.IVF;

public sealed class IndexIVFScalarQuantizer : CpuIndex<IndexIVFScalarQuantizer>, IFromNativeHandle<IndexIVFScalarQuantizer>
{
    private readonly INativeIndex _quantizer;

    public IndexIVFScalarQuantizer(INativeIndex quantizer, int dimensions, int nlist, QuantizerType qt, MetricType metric = MetricType.L2, bool encodeResidual = true)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexIVFScalarQuantizer_new_with_metric(out IntPtr ptr, quantizer.Handle, (UIntPtr)dimensions, (UIntPtr)nlist, qt, metric, encodeResidual)
        );
        
        SafeHandle = new FaissIndexHandle(ptr);
        
        Native.faiss_IndexIVFScalarQuantizer_set_own_fields(SafeHandle, false);
    }

    private IndexIVFScalarQuantizer(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexIVFScalarQuantizer_set_own_fields(SafeHandle, true);
        
        _quantizer = new GenericIndex(Native.faiss_IndexIVFScalarQuantizer_quantizer(SafeHandle));
    }
    
    static IndexIVFScalarQuantizer IFromNativeHandle<IndexIVFScalarQuantizer>.FromHandle(IntPtr handle) => new(handle);
    
    public int Nlist => (int)Native.faiss_IndexIVFScalarQuantizer_nlist(SafeHandle);
    
    public int Nprobe
    {
        get => (int)Native.faiss_IndexIVFScalarQuantizer_nprobe(SafeHandle);
        set => Native.faiss_IndexIVFScalarQuantizer_set_nprobe(SafeHandle, (UIntPtr)value);
    }
}