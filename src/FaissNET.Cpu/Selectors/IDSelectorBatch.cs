using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorBatch : IDSelector
{
    public IDSelectorBatch(ReadOnlySpan<long> indices) : base(CreateHandle(indices)) { }
    
    public int Nbits => Native.faiss_IDSelectorBatch_nbits(SafeHandle);

    public long Mask => Native.faiss_IDSelectorBatch_mask(SafeHandle);
    
    private static unsafe IntPtr CreateHandle(ReadOnlySpan<long> indices)
    {
        fixed (long* p = indices)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IDSelectorBatch_new(out IntPtr ptr, (UIntPtr)indices.Length, p)
            );
            
            return ptr;
        }
    }
}