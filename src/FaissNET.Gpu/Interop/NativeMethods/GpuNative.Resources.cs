using System.Runtime.InteropServices;
using Faiss.Gpu.Interop.SafeHandles;

namespace Faiss.Gpu.Interop.NativeMethods;

internal static partial class GpuNative
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_new(out IntPtr p_resources);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_StandardGpuResources_free(IntPtr resources);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_setTempMemory(GpuResourcesProviderHandle res, nuint size);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_noTempMemory(GpuResourcesProviderHandle res);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_setPinnedMemory(GpuResourcesProviderHandle res, nuint size);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_setDefaultStream(GpuResourcesProviderHandle res, int device, IntPtr stream);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(GpuResourcesProviderHandle res);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_initializeForDevice(IntPtr res, int device);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_syncDefaultStream(IntPtr res, int device);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_syncDefaultStreamCurrentDevice(IntPtr res);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuResources_free(IntPtr resources);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_GpuResourcesProvider_free(IntPtr provider);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResourcesProvider_getResources(GpuResourcesProviderHandle provider, out IntPtr res);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getBlasHandle(IntPtr res, int device, out IntPtr handle);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getDefaultStream(IntPtr res, int device, out IntPtr stream);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getPinnedMemory(IntPtr res, out IntPtr ptr, out nuint size);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getAsyncCopyStream(IntPtr res, int device, out IntPtr stream);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getBlasHandleCurrentDevice(IntPtr res, out IntPtr handle);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getDefaultStreamCurrentDevice(IntPtr res, out IntPtr stream);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_GpuResources_getAsyncCopyStreamCurrentDevice(IntPtr res, out IntPtr stream);
}