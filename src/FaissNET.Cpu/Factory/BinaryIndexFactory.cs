using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Factory;

public static class BinaryIndexFactory
{
    /// <summary>
    /// Creates a binary index from a string descriptor.
    /// </summary>
    /// <param name="description">Factory string (e.g., "BFlat", "BIVF256", "BHNSW32", "BHash16").</param>
    /// <param name="dimensions">Vector dimensionality in bits.</param>
    /// <param name="ownsHandle">If the created index should own its handle.</param>
    public static T Create<T>(string description, int dimensions, bool ownsHandle = true)
        where T : INativeBinaryIndex, IFromNativeBinaryIndexHandle<T>
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_binary_factory(out IntPtr ptr, dimensions, description)
        );

        return T.FromPointer(ptr, ownsHandle);
    }
}