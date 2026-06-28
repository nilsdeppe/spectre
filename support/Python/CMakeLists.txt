# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT ENABLE_PYTHON)
  return()
endif()

option(MACHINE "Select a machine that we know how to run on, such as a \
particular supercomputer" OFF)
spectre_python_add_module(
  support
  PYTHON_FILES
  CheckSpecImport.py
  CliExceptions.py
  DirectoryStructure.py
  Logging.py
  Machines.py
  Resubmit.py
  RunNext.py
  RunSlurmCommand.py
  Schedule.py
  Yaml.py
)

# Copy all machine definitions into the Python prefix so that we can identify
# the current machine from its hostname at runtime (see Machines.py).
file(GLOB _machine_yamls ${CMAKE_SOURCE_DIR}/support/Machines/*.yaml)
foreach(_machine_yaml ${_machine_yamls})
  get_filename_component(_machine_yaml_name ${_machine_yaml} NAME)
  configure_file(
    ${_machine_yaml}
    ${SPECTRE_PYTHON_PREFIX}/support/Machines/${_machine_yaml_name}
    COPYONLY
    )
endforeach()

if(MACHINE AND NOT MACHINE STREQUAL "UNKNOWN")
  message(STATUS "Selected machine: ${MACHINE}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/Machines/${MACHINE}.yaml
    ${SPECTRE_PYTHON_PREFIX}/support/Machine.yaml
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/SubmitTemplateBase.sh
    ${SPECTRE_PYTHON_PREFIX}/support/SubmitTemplateBase.sh
    @ONLY
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/${MACHINE}.sh
    ${SPECTRE_PYTHON_PREFIX}/support/SubmitTemplate.sh
    @ONLY
    )
endif()

# Generate shell completion scripts. These don't usually change, so they are
# committed to the source code and configured to the build dir. See the
# [click docs](https://click.palletsprojects.com/shell-completion/)
# for instructions how to regenerate the shell completion scripts.
configure_file(
  shell-completion.bash
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.bash @ONLY)
configure_file(
  shell-completion.fish
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.fish @ONLY)
configure_file(
  shell-completion.zsh
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.zsh @ONLY)
