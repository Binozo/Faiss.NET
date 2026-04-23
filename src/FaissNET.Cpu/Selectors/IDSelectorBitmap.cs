using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Selectors;

public sealed class IDSelectorBitmap : IDSelector
{
    public unsafe IDSelectorBitmap(ReadOnlySpan<byte> bitmap) : base(CreateHandle(bitmap)) { }
    
    private static unsafe IntPtr CreateHandle(ReadOnlySpan<byte> bitmap)
    {
        fixed (byte* p = bitmap)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_IDSelectorBitmap_new(out IntPtr ptr, (UIntPtr)bitmap.Length, p)
            );
            
            return ptr;
        }
    }
    
    public int N => (int)Native.faiss_IDSelectorBitmap_n(SafeHandle);
}