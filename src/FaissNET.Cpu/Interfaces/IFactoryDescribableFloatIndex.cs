using System.Runtime.InteropServices;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IFactoryDescribableFloatIndex : INativeIndex
{
    public string? GetFactoryDescription();
}

internal static class FactoryDescribableFloatIndexImpl
{
    public static unsafe string? GetFactoryDescription(INativeIndex index)
    {
        const int bufSize = 1024;
        byte* buf = stackalloc byte[bufSize];
        
        FaissErrorHandler.ThrowIfError(
            Native.faiss_reverse_index_factory(index.Handle, buf, bufSize)
        );
        
        var description = Marshal.PtrToStringAnsi((IntPtr)buf);
        if (string.IsNullOrEmpty(description)
            || description.StartsWith(',') || description.EndsWith(',')
            || description.Contains(",,") || description.Contains("()") 
            || description.Contains("(,") || description.Contains(",)"))
            return null;
        
        return description;
    }
}