using Microsoft.Extensions.VectorData;

namespace FaissNet.VectorData;

public sealed class FaissCollectionOptions : VectorStoreCollectionOptions
{
    public int BatchSize { get; set; } = 1024 * 10;
}