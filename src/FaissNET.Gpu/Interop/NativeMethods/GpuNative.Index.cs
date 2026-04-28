using System.Runtime.InteropServices;
using Faiss.Gpu.Interop.SafeHandles;
using Faiss.Interop.SafeHandles;

namespace Faiss.Gpu.Interop.NativeMethods;

internal static partial class GpuNative
{
    private const string LibraryName = "faiss_c";

    [LibraryImport(LibraryName)]
    internal static partial int faiss_index_cpu_to_gpu(GpuResourcesProviderHandle provider, int device, FaissIndexHandle index, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_index_cpu_to_gpu_with_options(GpuResourcesProviderHandle provider, int device, FaissIndexHandle index, GpuClonerOptionsHandle options, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_index_gpu_to_cpu(FaissIndexHandle gpu_index, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_index_cpu_to_gpu_multiple_with_options(IntPtr* providers_vec, nuint providers_vec_size, int* devices, nuint devices_size, FaissIndexHandle index, GpuMultipleClonerOptionsHandle options, out IntPtr p_out);

    [LibraryImport(LibraryName)]
    internal static unsafe partial int faiss_index_cpu_to_gpu_multiple(IntPtr* providers_vec, int* devices, nuint devices_size, FaissIndexHandle index, out IntPtr p_out);
}