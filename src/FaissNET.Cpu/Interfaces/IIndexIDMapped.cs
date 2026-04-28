namespace Faiss.Cpu.Interfaces;

/// <summary>
/// Represents an index that supports adding vectors with custom IDs.
/// </summary>
public interface IIndexIDMapped
{
    /// <summary>
    /// Adds vectors to the index with the specified IDs.
    /// </summary>
    /// <param name="count">The number of vectors to add.</param>
    /// <param name="vectors">The vectors to add.</param>
    /// <param name="xids">The IDs to assign to the vectors.</param>
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids);
}