using FaissNet.VectorData;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

namespace Microsoft.Extensions.DependencyInjection;

public static class FaissServiceCollectionExtensions
{
    public static IServiceCollection AddFaissVectorStore(this IServiceCollection services, FaissVectorStoreOptions? options = null, string? serviceId = null)
    {
        services.AddKeyedSingleton<FaissVectorStore>(serviceId, (provider, _) =>
            new FaissVectorStore(options ?? provider.GetService<FaissVectorStoreOptions>() ?? new FaissVectorStoreOptions
            {
                EmbeddingGenerator = provider.GetService<IEmbeddingGenerator>(),
            })
        );

        services.AddKeyedSingleton<VectorStore>(serviceId, (provider, _) => provider.GetRequiredKeyedService<FaissVectorStore>(serviceId));

        return services;
    }

    public static IServiceCollection AddFaissCollection<TKey, TRecord>(this IServiceCollection services, string name, FaissCollectionOptions? options = null, string? serviceId = null)
        where TKey : notnull
        where TRecord : class
    {
        services.AddKeyedSingleton<VectorStoreCollection<TKey, TRecord>>(serviceId, (provider, _) =>
        {
            var store = provider.GetRequiredKeyedService<FaissVectorStore>(serviceId);
            return new FaissCollection<TKey, TRecord>(store, name, options ?? new FaissCollectionOptions
            {
                EmbeddingGenerator = provider.GetService<IEmbeddingGenerator>(),
            });
        });

        services.AddKeyedSingleton<IVectorSearchable<TRecord>>(serviceId, (provider, _) => provider.GetRequiredKeyedService<VectorStoreCollection<TKey, TRecord>>(serviceId));

        return services;
    }
}