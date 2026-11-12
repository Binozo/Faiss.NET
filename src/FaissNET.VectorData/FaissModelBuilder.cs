using System;
using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace FaissNet.VectorData;

internal sealed class FaissModelBuilder() : CollectionModelBuilder(new CollectionModelBuildingOptions
{
    SupportsMultipleVectors = false,
    RequiresAtLeastOneVector = true,
})
{
    protected override void ValidateKeyProperty(KeyPropertyModel keyProperty)
    {
        base.ValidateKeyProperty(keyProperty); 

        Type type = keyProperty.Type;
        if (type != typeof(string) && type != typeof(Guid) && type != typeof(int) && type != typeof(long) && type != typeof(ulong))
            throw new NotSupportedException($"Key type '{type.Name}' is not supported. Supported key types: string, Guid, int, long, ulong.");
    }

    protected override bool IsVectorPropertyTypeValid(Type type, [NotNullWhen(false)] out string? supportedTypes)
    {
        supportedTypes = "ReadOnlyMemory<float>, Embedding<float>, float[]";
        type = Nullable.GetUnderlyingType(type) ?? type;

        return type == typeof(ReadOnlyMemory<float>)
               || type == typeof(Embedding<float>)
               || type == typeof(float[]);
    }

    protected override bool IsDataPropertyTypeValid(Type type, [NotNullWhen(false)] out string? supportedTypes)
    {
        supportedTypes = "string, int, long, double, bool, Guid, DateTimeOffset";
        type = Nullable.GetUnderlyingType(type) ?? type;

        return type == typeof(string) || type == typeof(int) || type == typeof(long) || type == typeof(double)
               || type == typeof(bool) || type == typeof(Guid) || type == typeof(DateTimeOffset);
    }
}