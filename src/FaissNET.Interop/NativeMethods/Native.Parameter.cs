using System.Runtime.InteropServices;
using Faiss.Interop.SafeHandles;

namespace Faiss.Interop.NativeMethods;

internal static unsafe partial class Native
{
    [LibraryImport(LibraryName)]
    internal static partial int faiss_ParameterSpace_new(out IntPtr pSpace);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_ParameterSpace_free(IntPtr space);

    [LibraryImport(LibraryName)]
    internal static partial UIntPtr faiss_ParameterSpace_n_combinations(FaissParameterSpaceHandle space);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ParameterSpace_combination_name(FaissParameterSpaceHandle space, UIntPtr cno, byte* buf, UIntPtr bufSize);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_ParameterSpace_set_index_parameters(FaissParameterSpaceHandle space, FaissIndexHandle index, string description);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ParameterSpace_set_index_parameters_cno(FaissParameterSpaceHandle space, FaissIndexHandle index, UIntPtr cno);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_ParameterSpace_set_index_parameters_binary(FaissParameterSpaceHandle space, FaissIndexBinaryHandle index, string description);

    [LibraryImport(LibraryName)]
    internal static partial int faiss_ParameterSpace_set_index_parameters_cno_binary(FaissParameterSpaceHandle space, FaissIndexBinaryHandle index, UIntPtr cno);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_ParameterSpace_set_index_parameter(FaissParameterSpaceHandle space, FaissIndexHandle index, string name, double value);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_ParameterSpace_set_index_parameter_binary(FaissParameterSpaceHandle space, FaissIndexBinaryHandle index, string name, double value);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_ParameterSpace_display(FaissParameterSpaceHandle space);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int faiss_ParameterSpace_add_range(FaissParameterSpaceHandle space, string name, out IntPtr pRange);

    [LibraryImport(LibraryName)]
    internal static partial IntPtr faiss_ParameterRange_name(IntPtr range);

    [LibraryImport(LibraryName)]
    internal static partial void faiss_ParameterRange_values(IntPtr range, out IntPtr values, out UIntPtr size);
}