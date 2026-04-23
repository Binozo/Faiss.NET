using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu;

public static class Faiss
{
    public static String Version => Native.FaissVersion;
}