using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Interfaces;

internal static class FloatDirectMapImpl
{
    public static void Make(FaissIndexHandle index, bool enable)
    {
        try
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexIVF_make_direct_map(index, enable));
        }
        catch (FaissNativeException)
        {
            // Reverse the failed map build
            Native.faiss_IndexIVF_make_direct_map(index, false);
            throw;
        }
    }
}

internal static class BinaryDirectMapImpl
{
    public static void Make(FaissBinaryIndexHandle index, bool enable)
    {
        try
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinaryIVF_make_direct_map(index, enable));
        }
        catch (FaissNativeException)
        {
            // Reverse the failed map build
            Native.faiss_IndexBinaryIVF_make_direct_map(index, false);
            throw;
        }
    }

    public static void SetType(FaissBinaryIndexHandle index, DirectMapType type)
    {
        try
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinaryIVF_set_direct_map_type(index, type));
        }
        catch (FaissNativeException)
        {
            // Reverse the failed map build
            Native.faiss_IndexBinaryIVF_make_direct_map(index, false);
            throw;
        }
    }
}
