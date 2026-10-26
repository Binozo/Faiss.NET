namespace Faiss.Tests.Infrastructure;

public interface ICapabilityCase
{
    bool Has(IndexCapabilities caps);
}

[Flags]
public enum IndexCapabilities
{
    None = 0,
    SequentialAdd = 1 << 0,
    Clone = 1 << 1,
    Serialize = 1 << 2,
    Reconstruct = 1 << 3,
    Train = 1 << 4,
    AddWithIds = 1 << 5,
    Codes = 1 << 6,
    Exact = 1 << 7,
    SelectorSearch = 1 << 8,
    RangeSearch = 1 << 9,
    LossyReconstruct = 1 << 10,
    ParamsSearch = 1 << 11,
    RemoveIds = 1 << 12,
    RemoveKeepsIds = 1 << 13,
    LossyCodes = 1 << 14,
    NearestReconstruction = 1 << 15,
}
