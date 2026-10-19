using System.Runtime.InteropServices;

namespace Faiss.Interop.SafeHandles;

/// <summary>
/// An owning handle to a native index that uses other native objects without owning them, such as an IndexIDMap
/// built with <c>takeOwnership: false</c> or an IndexShards built with <c>ownIndices: false</c>.
/// </summary>
/// <remarks>
/// Every registered dependency is kept alive until this handle has freed its own native object.
/// Tts handle cannot release while the reference taken here is outstanding. A caller disposing
/// a borrowed child early therefore only defers that free, instead of leaving this object
/// pointing at freed memory. This handle's own native object is freed first, so nothing native
/// can still be using a dependency once it is let go.
/// </remarks>
/// <typeparam name="TRelease">Native free function for the owned object.</typeparam>
internal sealed class FaissDependentIndexHandle<TRelease> : FaissIndexHandle where TRelease : struct, IFaissRelease
{
    private readonly List<SafeHandle> _dependencies = [];
    private readonly Lock _gate = new();

    internal FaissDependentIndexHandle(IntPtr preexistingHandle) : base(preexistingHandle, ownsHandle: true)
    {
    }

    /// <summary>
    /// Keeps <paramref name="dependency"/> alive until this handle is released.
    /// </summary>
    /// <exception cref="ObjectDisposedException">This handle or <paramref name="dependency"/> is already closed.</exception>
    internal void AddDependency(SafeHandle dependency)
    {
        ArgumentNullException.ThrowIfNull(dependency);

        var added = false;
        dependency.DangerousAddRef(ref added);

        lock (_gate)
        {
            if (!IsClosed)
            {
                _dependencies.Add(dependency);
                return;
            }
        }

        dependency.DangerousRelease();
        throw new ObjectDisposedException(GetType().Name);
    }

    /// <summary>
    /// Stops keeping one registration of <paramref name="dependency"/> alive, after native has dropped its reference.
    /// </summary>
    /// <returns><see langword="false"/> when <paramref name="dependency"/> was not registered.</returns>
    internal bool RemoveDependency(SafeHandle dependency)
    {
        ArgumentNullException.ThrowIfNull(dependency);

        lock (_gate)
        {
            if (!_dependencies.Remove(dependency))
            {
                return false;
            }
        }

        dependency.DangerousRelease();
        return true;
    }

    protected override bool ReleaseHandle()
    {
        // Free the owning object first to remove all dependencies to this handle.
        TRelease.Release(handle);

        SafeHandle[] dependencies;
        lock (_gate)
        {
            dependencies = [.. _dependencies];
            _dependencies.Clear();
        }

        foreach (var dependency in dependencies)
            dependency.DangerousRelease();

        return true;
    }
}
