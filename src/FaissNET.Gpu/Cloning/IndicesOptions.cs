namespace Faiss.Gpu.Cloning;

/// <summary>
/// Defines how user vector index data is stored on the GPU.
/// </summary>
public enum IndicesOptions
{
    /// <summary>
    /// Indices are stored only on the CPU. The GPU returns inverted-list
    /// offsets, which the CPU translates to real user indices.
    /// </summary>
    Cpu = 0,
    /// <summary>
    /// No indices are stored. Search reports each result as its inverted list in the upper 32 bits and its
    /// offset within that list in the lower 32 bits, and the index cannot be transferred back to the CPU.
    /// </summary>
    Ivf = 1,
    /// <summary>
    /// Indices are stored as 32-bit integers on the GPU. Suitable for
    /// collections with fewer than 2,147,483,648 vectors.
    /// </summary>
    Bit32 = 2,
    /// <summary>
    /// Indices are stored as 64-bit integers on the GPU.
    /// </summary>
    Bit64 = 3,
}