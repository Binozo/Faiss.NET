using System.Runtime.CompilerServices;
using Faiss.Interop;

namespace Faiss.Gpu.Interop;

/// <summary>
/// FaissNet.Gpu.dll carries its own [LibraryImport("faiss_c")] declarations, so it must
/// register the shared BLAS-flavor resolver too; otherwise this assembly would load the
/// default faiss_c while FaissNet.dll loads the MKL build (two faiss copies, mixed handles).
/// </summary>
internal static class GpuLibraryResolverInit
{
#pragma warning disable CA2255 // must register before any P/Invoke in this assembly; no app entry point exists in a library
    [ModuleInitializer]
    internal static void Init() =>
        FaissLibraryResolver.Register(typeof(GpuLibraryResolverInit).Assembly);
#pragma warning restore CA2255
}
