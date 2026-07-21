using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Faiss.Interop;

internal static class FaissLibraryResolver
{
    private const string LibraryName = "faiss_c";
    private const string MklLibraryName = "faiss_c.mkl";

    internal static string? LoadedFlavor { get; private set; }

    private static readonly Lazy<bool> PreferMkl = new(DetectPreferMkl);

#pragma warning disable CA2255 // must register before any P/Invoke in this assembly; no app entry point exists in a library
    [ModuleInitializer]
    internal static void Init() => Register(typeof(FaissLibraryResolver).Assembly);
#pragma warning restore CA2255

    internal static void Register(Assembly assembly) =>
        NativeLibrary.SetDllImportResolver(assembly, Resolve);

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LibraryName)
        {
            return IntPtr.Zero;
        }
        
        if (PreferMkl.Value && NativeLibrary.TryLoad(MklLibraryName, assembly, searchPath, out var handle))
        {
            LoadedFlavor = "mkl";
            return handle;
        }

        LoadedFlavor ??= "default";
        return IntPtr.Zero; // default (OpenBLAS) build
    }

    private static bool DetectPreferMkl()
    {
        switch (Environment.GetEnvironmentVariable("FAISSNET_BLAS")?.ToLowerInvariant())
        {
            case "mkl":
                return true;
            case "openblas" or "default":
                return false;
        }

        return IsGenuineIntel();
    }

    private static bool IsGenuineIntel()
    {
        if (!X86Base.IsSupported)
        {
            return false;
        }

        var (_, ebx, ecx, edx) = X86Base.CpuId(0, 0);
        // Vendor string "GenuineIntel": EBX = "Genu", EDX = "ineI", ECX = "ntel"
        return ebx == 0x756E6547 && edx == 0x49656E69 && ecx == 0x6C65746E;
    }
}
