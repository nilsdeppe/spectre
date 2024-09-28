# Distributed under the MIT License.
# See LICENSE.txt for details.

# Create an C++ standard library target for tracking dependencies
# through includes throughout SpECTRE.
if(NOT TARGET Stl)
  add_library(Stl INTERFACE IMPORTED)

  add_interface_lib_headers(
    TARGET Stl
    HEADERS
    algorithm
    any
    array
    atomic
    barrier
    bit
    bitset
    cassert
    cctype
    cerrno
    cfenv
    cfloat
    charconv
    chrono
    cinttypes
    climits
    clocale
    cmath
    codecvt
    compare
    complex
    concepts
    condition_variable
    coroutine
    csetjmp
    csignal
    cstdarg
    cstddef
    cstdint
    cstdio
    cstdlib
    cstring
    ctime
    cuchar
    cwchar
    cwctype
    deque
    exception
    execution
    format
    forward_list
    fstream
    functional
    future
    initializer_list
    iomanip
    ios
    iosfwd
    iostream
    istream
    iterator
    latch
    limits
    list
    locale
    map
    memory
    memory_resource
    mutex
    new
    numbers
    numeric
    optional
    ostream
    queue
    random
    ranges
    ratio
    regex
    scoped_allocator
    semaphore
    set
    shared_mutex
    source_location
    span
    sstream
    stack
    stdexcept
    stop_token
    streambuf
    string
    string_view
    strstream
    syncstream
    system_error
    thread
    tuple
    type_traits
    typeindex
    typeinfo
    unordered_map
    unordered_set
    utility
    valarray
    variant
    vector
    version

    # UNIX/Linux specific headers
    dirent.h
    emmintrin.h
    execinfo.h
    immintrin.h
    libgen.h
    link.h
    sys/stat.h
    sys/types.h
    unistd.h
    xmmintrin.h

    # C library
    assert.h
    complex.h
    ctype.h
    errorno.h
    fenv.h
    float.h
    inttypes.h
    iso646.h
    limits.h
    locale.h
    math.h
    setjmp.h
    signal.h
    stdalign.h
    stdarg.h
    stdatomic.h
    stdbit.h
    stdbool.h
    stdckdint.h
    stddef.h
    stdint.h
    stdio.h
    stdlib.h
    stdnoreturn.h
    string.h
    tgmath.h
    threads.h
    time.h
    uchar.h
    wchar.h
    wctype.h
    )

  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    Stl
    )
endif(NOT TARGET Stl)
