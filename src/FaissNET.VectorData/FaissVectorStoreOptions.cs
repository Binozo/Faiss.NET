using Microsoft.Extensions.AI;

namespace FaissNet.VectorData;

public sealed class FaissVectorStoreOptions
{
    public IEmbeddingGenerator? EmbeddingGenerator { get; set; }
}