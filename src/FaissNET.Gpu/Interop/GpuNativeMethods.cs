namespace Faiss.Gpu.Interop;

using System.Runtime.InteropServices;

internal static partial class GpuNativeMethods
{
    private const string LibraryName = "faiss_c";
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_new(out IntPtr p_resources);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_StandardGpuResources_free(IntPtr resources);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_setTempMemory(IntPtr resources, nuint size);
    
    [LibraryImport(LibraryName)]
    internal static partial int faiss_index_cpu_to_gpu(
        IntPtr provider,
        int device,
        IntPtr index,
        out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_index_gpu_to_cpu(
        IntPtr gpu_index,
        out IntPtr p_out);
}