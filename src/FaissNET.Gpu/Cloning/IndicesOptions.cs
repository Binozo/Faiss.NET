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
    /// No indices are stored. The GPU returns inverted-list offsets as the index.
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