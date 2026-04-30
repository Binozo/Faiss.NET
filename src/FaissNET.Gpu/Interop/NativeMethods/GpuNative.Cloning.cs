using System.Runtime.InteropServices;
using Faiss.Gpu.Interop.SafeHandles;

namespace Faiss.Gpu.Interop.NativeMethods;

internal static partial class GpuNative
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_new(out IntPtr options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_free(IntPtr options);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_useFloat16(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_useFloat16(GpuClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_indicesOptions(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_indicesOptions(GpuClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_usePrecomputed(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_usePrecomputed(GpuClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuMultipleClonerOptions_new(out IntPtr options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuMultipleClonerOptions_free(IntPtr options);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuMultipleClonerOptions_shard(GpuMultipleClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuMultipleClonerOptions_set_shard(GpuMultipleClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuMultipleClonerOptions_shard_type(GpuMultipleClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuMultipleClonerOptions_set_shard_type(GpuMultipleClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_useFloat16CoarseQuantizer(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_useFloat16CoarseQuantizer(GpuClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial CLong faiss_GpuClonerOptions_reserveVecs(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_reserveVecs(GpuClonerOptionsHandle options, CLong value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_storeTransposed(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_storeTransposed(GpuClonerOptionsHandle options, int value);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuClonerOptions_verbose(GpuClonerOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuClonerOptions_set_verbose(GpuClonerOptionsHandle options, int value);
}