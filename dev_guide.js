var dev_guide =
[
    [ "Notes on SpECTRE load-balancing using Charm++'s built-in load balancers", "load_balancing_notes.html", [
      [ "Charm++ Interface", "dev_guide.html#autotoc_md92", null ],
      [ "Continuous Integration", "dev_guide.html#autotoc_md93", null ],
      [ "CoordinateMap Guide", "dev_guide.html#autotoc_md94", null ],
      [ "Developing and Improving Executables", "dev_guide.html#autotoc_md95", null ],
      [ "Foundational Concepts in SpECTRE", "dev_guide.html#autotoc_md96", null ],
      [ "General SpECTRE Terminology", "dev_guide.html#autotoc_md97", null ],
      [ "Having your Contributions Merged into SpECTRE", "dev_guide.html#autotoc_md98", null ],
      [ "Performance and Optimization", "dev_guide.html#autotoc_md99", null ],
      [ "Technical Documentation for Fluent Developers", "dev_guide.html#autotoc_md100", null ],
      [ "Template Metaprogramming (TMP)", "dev_guide.html#autotoc_md101", [
        [ "Overview of how LBs work with SpECTRE", "load_balancing_notes.html#autotoc_md110", null ],
        [ "Contaminated LB measurements on first invocation", "load_balancing_notes.html#autotoc_md111", null ],
        [ "Scotch load balancer", "load_balancing_notes.html#autotoc_md112", null ],
        [ "General recommendations", "load_balancing_notes.html#autotoc_md113", [
          [ "Homogeneous loads", "load_balancing_notes.html#autotoc_md114", null ],
          [ "Inhomogeneous loads", "load_balancing_notes.html#autotoc_md115", null ]
        ] ]
      ] ]
    ] ],
    [ "Automatic versioning", "dev_guide_automatic_versioning.html", [
      [ "Creating releases", "dev_guide_automatic_versioning.html#autotoc_md74", null ],
      [ "Release notes", "dev_guide_automatic_versioning.html#autotoc_md75", null ]
    ] ],
    [ "Redistributing Gridpoints", "redistributing_gridpoints.html", [
      [ "Introduction", "redistributing_gridpoints.html#autotoc_md162", null ],
      [ "Generalized Logical Coordinates", "redistributing_gridpoints.html#autotoc_md163", null ],
      [ "Equiangular Maps", "redistributing_gridpoints.html#autotoc_md164", null ],
      [ "Projective Maps", "redistributing_gridpoints.html#autotoc_md165", null ]
    ] ],
    [ "Build System", "spectre_build_system.html", [
      [ "CMake", "spectre_build_system.html#cmake", [
        [ "Adding Source Files", "spectre_build_system.html#adding_source_files", [
          [ "Adding Libraries", "spectre_build_system.html#adding_libraries", null ]
        ] ],
        [ "Adding Unit Tests", "spectre_build_system.html#adding_unit_tests", null ],
        [ "Adding Executables", "spectre_build_system.html#adding_executables", null ],
        [ "Adding External Dependencies", "spectre_build_system.html#adding_external_dependencies", null ],
        [ "Commonly Used CMake flags", "spectre_build_system.html#common_cmake_flags", null ],
        [ "CMake targets", "spectre_build_system.html#autotoc_md83", null ],
        [ "Checking Dependencies", "spectre_build_system.html#autotoc_md84", null ],
        [ "Formaline", "spectre_build_system.html#autotoc_md85", null ]
      ] ]
    ] ],
    [ "Creating Executables", "dev_guide_creating_executables.html", null ],
    [ "Writing SpECTRE executables", "tutorials_parallel.html", "tutorials_parallel" ],
    [ "Option Parsing", "dev_guide_option_parsing.html", [
      [ "Metadata and options", "dev_guide_option_parsing.html#autotoc_md120", null ],
      [ "General option format", "dev_guide_option_parsing.html#autotoc_md121", null ],
      [ "Constructible classes", "dev_guide_option_parsing.html#autotoc_md122", null ],
      [ "Factory", "dev_guide_option_parsing.html#autotoc_md123", null ],
      [ "Custom parsing", "dev_guide_option_parsing.html#autotoc_md124", null ]
    ] ],
    [ "Importing data", "dev_guide_importing.html", [
      [ "Importing volume data", "dev_guide_importing.html#autotoc_md109", null ]
    ] ],
    [ "Profiling", "profiling.html", [
      [ "Profiling with HPCToolkit", "profiling.html#profiling_with_hpctoolkit", null ],
      [ "Profiling with AMD uProf", "profiling.html#profiling_with_amd_uprof", null ],
      [ "Profiling With Charm++ Projections", "profiling.html#profiling_with_projections", [
        [ "Running SpECTRE With Trace Output", "profiling.html#autotoc_md141", null ],
        [ "Visualizing Trace %Data In Projections", "profiling.html#autotoc_md142", null ]
      ] ]
    ] ],
    [ "Writing Python Bindings", "spectre_writing_python_bindings.html", [
      [ "CMake and Directory Layout", "spectre_writing_python_bindings.html#autotoc_md143", null ],
      [ "Writing Bindings", "spectre_writing_python_bindings.html#autotoc_md144", null ],
      [ "Testing Python Bindings and Code", "spectre_writing_python_bindings.html#autotoc_md145", null ],
      [ "Using The Bindings", "spectre_writing_python_bindings.html#autotoc_md146", null ],
      [ "Notes:", "spectre_writing_python_bindings.html#autotoc_md147", null ],
      [ "Guidelines for writing command-line interfaces (CLIs)", "spectre_writing_python_bindings.html#autotoc_md148", null ]
    ] ],
    [ "Implementing SpECTRE vectors", "implementing_vectors.html", [
      [ "Overview of SpECTRE Vectors", "implementing_vectors.html#general_structure", null ],
      [ "The class definition", "implementing_vectors.html#class_definition", null ],
      [ "Allowed operator specification", "implementing_vectors.html#blaze_definitions", null ],
      [ "Supporting operations for <tt>std::array</tt>s of vectors", "implementing_vectors.html#array_vector_definitions", null ],
      [ "Equivalence operators", "implementing_vectors.html#Vector_type_equivalence", null ],
      [ "MakeWithValueImpl", "implementing_vectors.html#Vector_MakeWithValueImpl", null ],
      [ "Interoperability with other data types", "implementing_vectors.html#Vector_tensor_and_variables", null ],
      [ "Writing tests", "implementing_vectors.html#Vector_tests", [
        [ "Utility check functions", "implementing_vectors.html#autotoc_md102", [
          [ "<tt>TestHelpers::VectorImpl::vector_test_construct_and_assign()</tt>", "implementing_vectors.html#autotoc_md103", null ],
          [ "<tt>TestHelpers::VectorImpl::vector_test_serialize()</tt>", "implementing_vectors.html#autotoc_md104", null ],
          [ "<tt>TestHelpers::VectorImpl::vector_test_ref()</tt>", "implementing_vectors.html#autotoc_md105", null ],
          [ "<tt>TestHelpers::VectorImpl::vector_test_math_after_move()</tt>", "implementing_vectors.html#autotoc_md106", null ],
          [ "<tt>TestHelpers::VectorImpl::vector_ref_test_size_error()</tt>", "implementing_vectors.html#autotoc_md107", null ]
        ] ],
        [ "<tt>TestHelpers::VectorImpl::test_functions_with_vector_arguments()</tt>", "implementing_vectors.html#autotoc_md108", null ]
      ] ],
      [ "Vector storage nuts and bolts", "implementing_vectors.html#Vector_storage", null ]
    ] ],
    [ "Understanding Compiler and Linker Errors", "compiler_and_linker_errors.html", [
      [ "Linker Errors", "compiler_and_linker_errors.html#understanding_linker_errors", null ]
    ] ],
    [ "Static Analysis Tools", "static_analysis_tools.html", null ],
    [ "Build Profiling and Optimization", "build_profiling_and_optimization.html", [
      [ "Why is our build so expensive?", "build_profiling_and_optimization.html#autotoc_md76", null ],
      [ "Understanding template expenses", "build_profiling_and_optimization.html#autotoc_md77", null ],
      [ "Profiling the build", "build_profiling_and_optimization.html#autotoc_md78", [
        [ "Specialized tests and feature exploration", "build_profiling_and_optimization.html#autotoc_md79", null ],
        [ "Templight++", "build_profiling_and_optimization.html#autotoc_md80", null ],
        [ "Clang profiling", "build_profiling_and_optimization.html#autotoc_md81", null ],
        [ "Clang AST syntax generation", "build_profiling_and_optimization.html#autotoc_md82", null ]
      ] ]
    ] ],
    [ "Tips for debugging an executable", "runtime_errors.html", [
      [ "Useful gdb commands", "runtime_errors.html#autotoc_md91", null ]
    ] ],
    [ "Motivation for SpECTRE's DataBox", "databox_foundations.html", [
      [ "Introduction", "databox_foundations.html#databox_introduction", null ],
      [ "Towards SpECTRE's DataBox", "databox_foundations.html#databox_towards_spectres_databox", [
        [ "Working without DataBoxes", "databox_foundations.html#databox_working_without_databoxes", null ],
        [ "A std::map DataBox", "databox_foundations.html#databox_a_std_map_databox", null ],
        [ "A std::tuple DataBox", "databox_foundations.html#databox_a_std_tuple_databox", null ],
        [ "A TaggedTuple DataBox", "databox_foundations.html#databox_a_taggedtuple_databox", null ]
      ] ],
      [ "SpECTRE's DataBox", "databox_foundations.html#databox_a_proper_databox", [
        [ "SimpleTags", "databox_foundations.html#databox_documentation_for_simple_tags", null ],
        [ "ComputeTags", "databox_foundations.html#databox_documentation_for_compute_tags", null ],
        [ "Mutating DataBox items", "databox_foundations.html#databox_documentation_for_mutate_tags", null ]
      ] ],
      [ "Toward SpECTRE's Actions", "databox_foundations.html#databox_towards_actions", [
        [ "Mutators", "databox_foundations.html#databox_documentation_for_mutators", null ]
      ] ]
    ] ],
    [ "Protocols", "protocols.html", [
      [ "Overview of protocols", "protocols.html#protocols_overview", null ],
      [ "Protocol users: Conforming to a protocol", "protocols.html#protocols_conforming", null ],
      [ "Protocol authors: Writing a protocol", "protocols.html#protocols_author", null ],
      [ "Protocol authors: Testing a protocol", "protocols.html#protocols_testing", null ],
      [ "Protocols and C++20 \"Constraints and concepts\"", "protocols.html#protocols_and_constraints", null ]
    ] ],
    [ "Domain Concepts", "domain_concepts.html", null ],
    [ "Writing Good Documentation", "writing_good_dox.html", [
      [ "Tutorials, Instructions, and Dev Guide", "writing_good_dox.html#writing_dox_writing_help", null ],
      [ "C++ Documentation", "writing_good_dox.html#writing_dox_cpp_dox_help", [
        [ "Add your object to an existing Module:", "writing_good_dox.html#autotoc_md166", null ],
        [ "Add a new Module:", "writing_good_dox.html#autotoc_md167", null ],
        [ "Add a new namespace:", "writing_good_dox.html#autotoc_md168", null ],
        [ "Put any mathematical expressions in your documentation into LaTeX form:", "writing_good_dox.html#autotoc_md169", null ],
        [ "Cite publications in your documentation", "writing_good_dox.html#writing_dox_citations", null ],
        [ "Include any pictures that aid in the understanding of your documentation:", "writing_good_dox.html#autotoc_md170", null ]
      ] ],
      [ "Python Documentation", "writing_good_dox.html#writing_dox_python_dox_help", null ]
    ] ],
    [ "Writing Unit Tests", "writing_unit_tests.html", [
      [ "Input file tests", "writing_unit_tests.html#autotoc_md176", null ]
    ] ],
    [ "GitHub Actions Continuous Integration", "github_actions_guide.html", [
      [ "Testing SpECTRE with GitHub Actions CI", "github_actions_guide.html#github_actions_ci", [
        [ "What is tested", "github_actions_guide.html#what-is-tested", null ],
        [ "How to perform the checks locally", "github_actions_guide.html#perform-checks-locally", null ],
        [ "Troubleshooting", "github_actions_guide.html#github-actions-troubleshooting", null ],
        [ "Precompiled Headers and ccache", "github_actions_guide.html#precompiled-headers-ccache", null ],
        [ "Caching Dependencies on macOS Builds", "github_actions_guide.html#caching-mac-os", null ]
      ] ]
    ] ],
    [ "Code Review Guide", "code_review_guide.html", null ],
    [ "General Performance Guidelines", "general_perf_guide.html", null ],
    [ "Observers Infrastructure", "observers_infrastructure_dev_guide.html", null ],
    [ "Parallelization, Charm++, and Core Concepts", "dev_guide_parallelization_foundations.html", [
      [ "Introduction", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_introduction", null ],
      [ "The Metavariables Class", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_metavariables_class", null ],
      [ "Phases of an Execution", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_phases_of_execution", null ],
      [ "The Algorithm", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_core_algorithm", null ],
      [ "Parallel Components", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_parallel_components", [
        [ "1. Types of Parallel Components", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_component_types", null ],
        [ "2. Requirements", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_component_requirements", null ],
        [ "3. Examples", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_component_examples", null ],
        [ "4. Placement", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_component_placement", null ]
      ] ],
      [ "Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_actions", [
        [ "1. Simple Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_simple_actions", null ],
        [ "2. Iterable Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_iterable_actions", null ],
        [ "3. Reduction Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_reduction_actions", null ],
        [ "4. Threaded Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_threaded_actions", null ],
        [ "5. Local Synchronous Actions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_local_synchronous_actions", null ]
      ] ],
      [ "Mutable items in the GlobalCache", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_mutable_global_cache", [
        [ "1. Specification of mutable GlobalCache items", "dev_guide_parallelization_foundations.html#autotoc_md136", null ],
        [ "2. Use of mutable GlobalCache items", "dev_guide_parallelization_foundations.html#autotoc_md137", [
          [ "1. Checking if the item is up-to-date", "dev_guide_parallelization_foundations.html#autotoc_md138", null ],
          [ "2. Retrieving the item", "dev_guide_parallelization_foundations.html#autotoc_md139", null ]
        ] ],
        [ "3. Modifying a mutable GlobalCache item", "dev_guide_parallelization_foundations.html#autotoc_md140", null ]
      ] ],
      [ "Charm++ Node and Processor Level Initialization Functions", "dev_guide_parallelization_foundations.html#dev_guide_parallelization_charm_node_processor_level_initialization", null ]
    ] ],
    [ "SFINAE", "sfinae.html", null ],
    [ "Metaprogramming with Brigand", "brigand.html", [
      [ "Metafunctions", "brigand.html#Metafunctions", [
        [ "Eager and lazy metafunctions", "brigand.html#lazy", null ],
        [ "Brigand metalambdas", "brigand.html#metalambdas", null ],
        [ "Evaluation of metalambdas", "brigand.html#metalambda_structure", [
          [ "Argument", "brigand.html#args", null ],
          [ "Lazy expression", "brigand.html#metalambda_lazy", null ],
          [ "Bind expression", "brigand.html#bind", null ],
          [ "Pin expression", "brigand.html#pin", null ],
          [ "Defer expression", "brigand.html#defer", null ],
          [ "Parent expression", "brigand.html#parent", null ],
          [ "Constant", "brigand.html#metalambda_constant", null ],
          [ "Metaclosure", "brigand.html#metalambda_metaclosure", null ]
        ] ],
        [ "Examples", "brigand.html#Examples", [
          [ "evens", "brigand.html#evens", null ],
          [ "maybe_first", "brigand.html#maybe_first", null ],
          [ "factorial", "brigand.html#factorial", null ],
          [ "make_subtracter", "brigand.html#make_subtracter", null ],
          [ "multiplication_table", "brigand.html#multiplication_table", null ],
          [ "column_with_zeros", "brigand.html#column_with_zeros", null ],
          [ "factorial_recursion", "brigand.html#factorial_recursion", null ],
          [ "primes", "brigand.html#primes", null ]
        ] ]
      ] ],
      [ "Guidelines for writing metafunctions", "brigand.html#metafunction_guidelines", null ],
      [ "Brigand types and functions", "brigand.html#function_docs", [
        [ "Containers", "brigand.html#Containers", [
          [ "integral_constant<T, value>", "brigand.html#integral_constant", null ],
          [ "list<T...>", "brigand.html#list", null ],
          [ "map<Pair...>", "brigand.html#map", null ],
          [ "pair<T1, T2>", "brigand.html#pair", null ],
          [ "set<T...>", "brigand.html#set", null ],
          [ "type_<T>", "brigand.html#type_", null ]
        ] ],
        [ "Constants", "brigand.html#Constants", [
          [ "empty_base", "brigand.html#empty_base", null ],
          [ "empty_sequence", "brigand.html#empty_sequence", null ],
          [ "false_type", "brigand.html#false_type", null ],
          [ "no_such_type_", "brigand.html#no_such_type_", null ],
          [ "true_type", "brigand.html#true_type", null ]
        ] ],
        [ "Constructor-like functions for lists", "brigand.html#list_constructor", [
          [ "filled_list<Entry, n, [Head]>", "brigand.html#filled_list", null ],
          [ "integral_list<T, n...>", "brigand.html#integral_list", null ],
          [ "make_sequence<Start, n, [Next], [Head]>", "brigand.html#make_sequence", null ],
          [ "range<T, start, stop>", "brigand.html#range", null ],
          [ "reverse_range<T, start, stop>", "brigand.html#reverse_range", null ]
        ] ],
        [ "Functions for querying lists", "brigand.html#list_query", [
          [ "all<Sequence, [Predicate]>", "brigand.html#all", null ],
          [ "any<Sequence, [Predicate]>", "brigand.html#any", null ],
          [ "at<Sequence, Index>", "brigand.html#at", null ],
          [ "at_c<Sequence, n>", "brigand.html#at_c", null ],
          [ "back<Sequence>", "brigand.html#back", null ],
          [ "count_if<Sequence, Predicate>", "brigand.html#count_if", null ],
          [ "fold<Sequence, State, Functor>", "brigand.html#fold", null ],
          [ "found<Sequence, [Predicate]>", "brigand.html#found", null ],
          [ "front<Sequence>", "brigand.html#front", null ],
          [ "<Sequence, Predicate, [NotFound]>", "brigand.html#index_if", null ],
          [ "index_of<Sequence, T>", "brigand.html#index_of", null ],
          [ "list_contains<Sequence, T>", "brigand.html#list_contains", null ],
          [ "none<Sequence, [Predicate]>", "brigand.html#none", null ],
          [ "not_found<Sequence, [Predicate]>", "brigand.html#not_found", null ],
          [ "size<Sequence>", "brigand.html#size", null ]
        ] ],
        [ "Functions producing lists from other lists", "brigand.html#list_to_list", [
          [ "append<Sequence...>", "brigand.html#append", null ],
          [ "clear<Sequence>", "brigand.html#clear", null ],
          [ "erase<Sequence, Index>", "brigand.html#erase", null ],
          [ "erase_c<Sequence, n>", "brigand.html#erase_c", null ],
          [ "filter<Sequence, Predicate>", "brigand.html#filter", null ],
          [ "find<Sequence, [Predicate]>", "brigand.html#find", null ],
          [ "flatten<Sequence>", "brigand.html#flatten", null ],
          [ "join<Sequence>", "brigand.html#join", null ],
          [ "list_difference<Sequence1, Sequence2>", "brigand.html#list_difference", null ],
          [ "merge<Sequence1, Sequence2, [Comparator]>", "brigand.html#merge", null ],
          [ "partition<Sequence, Predicate>", "brigand.html#partition", null ],
          [ "pop_back<Sequence, [Count]>", "brigand.html#pop_back", null ],
          [ "pop_front<Sequence, [Count]>", "brigand.html#pop_front", null ],
          [ "push_back<Sequence, T...>", "brigand.html#push_back", null ],
          [ "push_front<Sequence, T...>", "brigand.html#push_front", null ],
          [ "remove<Sequence, T>", "brigand.html#remove", null ],
          [ "remove_duplicates<Sequence>", "brigand.html#remove_duplicates", null ],
          [ "remove_if<Sequence, Predicate>", "brigand.html#remove_if", null ],
          [ "replace<Sequence, Old, New>", "brigand.html#replace", null ],
          [ "replace_if<Sequence, Predicate, T>", "brigand.html#replace_if", null ],
          [ "reverse<Sequence>", "brigand.html#reverse", null ],
          [ "reverse_find<Sequence, [Predicate]>", "brigand.html#reverse_find", null ],
          [ "reverse_fold<Sequence, State, Functor>", "brigand.html#reverse_fold", null ],
          [ "sort<Sequence, [Comparator]>", "brigand.html#sort", null ],
          [ "split<Sequence, Delimiter>", "brigand.html#split", null ],
          [ "split_at<Sequence, Index>", "brigand.html#split_at", null ],
          [ "transform<Sequence, Sequences..., Functor>", "brigand.html#transform", null ]
        ] ],
        [ "Operations on maps", "brigand.html#map_operations", [
          [ "at<Map, Key>", "brigand.html#map_at", null ],
          [ "erase<Map, Key>", "brigand.html#map_erase", null ],
          [ "has_key<Map, Key>", "brigand.html#map_has_key", null ],
          [ "insert<Map, Pair>", "brigand.html#map_insert", null ],
          [ "keys_as_sequence<Map, [Head]>", "brigand.html#keys_as_sequence", null ],
          [ "lookup<Map, Key>", "brigand.html#lookup", null ],
          [ "lookup_at<Map, Key>", "brigand.html#lookup_at", null ],
          [ "values_as_sequence<Map, [Head]>", "brigand.html#values_as_sequence", null ]
        ] ],
        [ "Operations on sets", "brigand.html#set_operations", [
          [ "contains<Set, T>", "brigand.html#contains", null ],
          [ "erase<Set, T>", "brigand.html#set_erase", null ],
          [ "has_key<Set, T>", "brigand.html#set_has_key", null ],
          [ "insert<Set, T>", "brigand.html#set_insert", null ]
        ] ],
        [ "Mathematical functions", "brigand.html#math", [
          [ "Arithmetic operators", "brigand.html#math_arithmetic", null ],
          [ "Bitwise operators", "brigand.html#math_bitwise", null ],
          [ "Comparison operators", "brigand.html#math_comparison", null ],
          [ "Logical operators", "brigand.html#math_logical", null ],
          [ "identity<T>", "brigand.html#identity", null ],
          [ "max<T1, T2>", "brigand.html#max", null ],
          [ "min<T1, T2>", "brigand.html#min", null ],
          [ "next<T>", "brigand.html#next", null ],
          [ "prev<T>", "brigand.html#prev", null ]
        ] ],
        [ "Miscellaneous functions", "brigand.html#misc", [
          [ "always<T>", "brigand.html#always", null ],
          [ "apply<Lambda, [Arguments...]>", "brigand.html#apply", null ],
          [ "count<T...>", "brigand.html#count", null ],
          [ "conditional_t<b, TrueResult, FalseResult>", "brigand.html#conditional_t", null ],
          [ "eval_if<Condition, TrueFunction, FalseFunction>", "brigand.html#eval_if", null ],
          [ "eval_if_c<b, TrueFunction, FalseFunction>", "brigand.html#eval_if_c", null ],
          [ "has_type<Ignored, [T]>", "brigand.html#has_type", null ],
          [ "if_<Condition, TrueResult, FalseResult>", "brigand.html#if_", null ],
          [ "if_c<Condition, TrueResult, FalseResult>", "brigand.html#if_c", null ],
          [ "inherit<T...>", "brigand.html#inherit", null ],
          [ "inherit_linearly<Sequence, NodePattern, [Root]>", "brigand.html#inherit_linearly", null ],
          [ "is_set<T...>", "brigand.html#is_set", null ],
          [ "real_<RealType, IntType, value>", "brigand.html#real_", null ],
          [ "repeat<Function, Count, Initial>", "brigand.html#repeat", null ],
          [ "sizeof_<T>", "brigand.html#sizeof_", null ],
          [ "substitute<Pattern, ArgumentList>", "brigand.html#substitute", null ],
          [ "type_from<T>", "brigand.html#type_from", null ],
          [ "wrap<Sequence, Head>", "brigand.html#wrap", null ]
        ] ],
        [ "Runtime functionality", "brigand.html#runtime", [
          [ "for_each_args(functor, arguments...)", "brigand.html#for_each_args", null ],
          [ "for_each<Sequence>(functor)", "brigand.html#for_each", null ],
          [ "select<Condition>(true_result, false_result)", "brigand.html#select", null ]
        ] ],
        [ "External integration", "brigand.html#external", [
          [ "Boost", "brigand.html#boost_integration", null ],
          [ "STL", "brigand.html#stl_integration", null ],
          [ "integral_constant", "brigand.html#make_integral", null ],
          [ "list", "brigand.html#as_list", null ],
          [ "set", "brigand.html#as_set", null ]
        ] ]
      ] ],
      [ "Bugs/Oddities", "brigand.html#oddities", null ],
      [ "TODO", "brigand.html#TODO", null ]
    ] ]
];