namespace Faiss.Interfaces;

/// <inheritdoc/>
public interface ITrainableBinaryIndex : IBinaryIndex
{
    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    bool IsTrained { get; }
    
    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="count">Number of training vectors (should be at least 10x the number of centroids).</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    Task TrainAsync(long count, ReadOnlyMemory<byte> vectors);
}