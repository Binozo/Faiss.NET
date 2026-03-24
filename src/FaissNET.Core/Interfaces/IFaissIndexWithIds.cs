namespace Faiss.Interfaces;

/// <summary>
/// Faiss index with custom database IDs support.
/// </summary>
public interface IFaissIndexWithIds : IFaissIndex
{
    /// <summary>
    /// Adds vectors to the index using your own custom IDs instead of Faiss's standard sequential ones.
    /// </summary>
    /// <param name="count">The number of vectors you are adding.</param>
    /// <param name="vectors">The flat span of vector data.</param>
    /// <param name="xids">Your custom database IDs that map exactly to the vectors.</param>
    void AddWithIds(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids);
}