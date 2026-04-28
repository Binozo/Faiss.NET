using System.Runtime.InteropServices;

namespace Faiss.Gpu.Interop.NativeMethods;

internal static partial class GpuNative
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_get_num_gpus(out int p_num_gpus);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_gpu_sync_all_devices();

    [LibraryImport(LibraryName)]
    internal static partial int faiss_gpu_profiler_start();

    [LibraryImport(LibraryName)]
    internal static partial int faiss_gpu_profiler_stop();
}