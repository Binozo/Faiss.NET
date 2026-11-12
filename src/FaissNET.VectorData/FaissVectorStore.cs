using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

namespace FaissNet.VectorData;

public sealed class FaissVectorStore : VectorStore
{
    private readonly ConcurrentDictionary<string, FaissCollectionState> _collections = new();
    private readonly Lock _createLock = new();
    
    private readonly FaissVectorStoreOptions? _options;
    internal IEmbeddingGenerator? DefaultEmbeddingGenerator => _options?.EmbeddingGenerator;

    private readonly VectorStoreMetadata _metadata = new()
    {
        VectorStoreSystemName = "faiss",
    };

    public FaissVectorStore(FaissVectorStoreOptions? options = null)
    {
        _options = options;
    }
    
    public override VectorStoreCollection<TKey, TRecord> GetCollection<TKey, TRecord>(string name, VectorStoreCollectionDefinition? definition = null)
    {
        if (typeof(TRecord) == typeof(Dictionary<string, object?>))
            throw new ArgumentException($"Use {nameof(GetDynamicCollection)} for Dictionary<string, object?> records.");

        return new FaissCollection<TKey, TRecord>(this, name, new FaissCollectionOptions
        {
            Definition = definition,
            EmbeddingGenerator = _options?.EmbeddingGenerator,
        });
    }

    public override VectorStoreCollection<object, Dictionary<string, object?>> GetDynamicCollection(string name, VectorStoreCollectionDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);

        return new FaissCollection<object, Dictionary<string, object?>>(this, name, new FaissCollectionOptions
        {
            Definition = definition,
            EmbeddingGenerator = _options?.EmbeddingGenerator,
        });
    }

    public override Task<bool> CollectionExistsAsync(string name, CancellationToken cancellationToken = default)
        => Task.FromResult(_collections.ContainsKey(name));

    public override async IAsyncEnumerable<string> ListCollectionNamesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;

        foreach (var name in _collections.Keys)
            yield return name;
    }

    public override Task EnsureCollectionDeletedAsync(string name, CancellationToken cancellationToken = default)
    {
        RemoveState(name);
        return Task.CompletedTask;
    }

    internal bool TryGetState(string name, [NotNullWhen(true)] out FaissCollectionState? state)
        => _collections.TryGetValue(name, out state);

    internal FaissCollectionState GetOrAddState(string name, Func<FaissCollectionState> create)
    {
        if (_collections.TryGetValue(name, out var existing))
            return existing;

        lock (_createLock)
        {
            return _collections.GetOrAdd(name, _ => create());
        }
    }

    internal void RemoveState(string name)
    {
        if (_collections.TryRemove(name, out var state))
            state.Dispose();
    }

    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        if (serviceKey is not null)
            return null;

        if (serviceType == typeof(VectorStoreMetadata))
            return _metadata;

        return serviceType.IsInstanceOfType(this)
            ? this
            : null;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            foreach (var name in _collections.Keys)
                RemoveState(name);
    }
}