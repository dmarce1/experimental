#!/usr/bin/env bash
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

set -u

showUsage() {
    printf 'Usage: %s\n' "${0##*/}"
    printf 'Run the interactive CosmicTiger configure and build menu in a terminal.\n'
}

case "${1:-}" in
    --help|-h)
        showUsage
        exit 0
        ;;
    '') ;;
    *)
        showUsage >&2
        exit 2
        ;;
esac

if [[ ! -t 0 || ! -t 1 ]]; then
    printf 'An interactive terminal is required. Run ./build.sh in a terminal.\n' >&2
    exit 2
fi

rootDir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 1
cd -- "$rootDir" || exit 1

coreCount=
if command -v nproc >/dev/null 2>&1; then
    # OpenMP thread settings can make nproc report 1 even when more CPUs are available.
    coreCount=$(unset OMP_NUM_THREADS OMP_THREAD_LIMIT; nproc)
fi
if [[ ! $coreCount =~ ^[1-9][0-9]*$ ]] && command -v getconf >/dev/null 2>&1; then
    coreCount=$(getconf _NPROCESSORS_ONLN)
fi
if [[ ! $coreCount =~ ^[1-9][0-9]*$ ]]; then
    coreCount=1
fi

buildMode=release
enableCuda=ON
testSuite=OFF
buildJobs=4
testJobs=2
if (( buildJobs > coreCount )); then
    buildJobs=$coreCount
fi
if (( testJobs > coreCount )); then
    testJobs=$coreCount
fi
hpxPrefix=
cudaArchitectures=
logFile=
promptValue=

trap 'printf "\nInterrupted. Log: %s\n" "${logFile:-none}" >&2; exit 130' INT
trap 'printf "\nTerminated. Log: %s\n" "${logFile:-none}" >&2; exit 143' TERM

readInput() {
    printf '%s' "$1"
    if ! IFS= read -r promptValue; then
        printf '\nInput closed; no build started.\n'
        exit 0
    fi
}

chooseMode() {
    while true; do
        printf '\nBuild mode: 1) Release  2) RelWithDebInfo  3) Debug  4) Profile\n'
        readInput "Choice [${buildMode}]: "
        case "${promptValue,,}" in
            '') return ;;
            1|release) buildMode=release; return ;;
            2|relwithdebinfo) buildMode=relwithdebinfo; return ;;
            3|debug) buildMode=debug; return ;;
            4|profile) buildMode=profile; return ;;
        esac
        printf 'Choose 1 through 4, or press Enter to keep the current mode.\n'
    done
}

chooseToggle() {
    local label=$1
    local currentValue=$2
    while true; do
        readInput "$label [${currentValue}] (y/n, Enter keeps): "
        case "${promptValue,,}" in
            '') promptValue=$currentValue; return ;;
            y|yes|on) promptValue=ON; return ;;
            n|no|off) promptValue=OFF; return ;;
        esac
        printf 'Enter y or n.\n'
    done
}

chooseNumber() {
    local label=$1
    local currentValue=$2
    local minimum=$3
    local maximum=$4
    local numericValue
    while true; do
        readInput "$label [${currentValue}] (${minimum}-${maximum}, Enter keeps): "
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && ${#promptValue} -le 6 ]]; then
            numericValue=$((10#$promptValue))
            if (( numericValue >= minimum && numericValue <= maximum )); then
                promptValue=$numericValue
                return
            fi
        fi
        printf 'Enter a whole number from %s to %s.\n' "$minimum" "$maximum"
    done
}

chooseText() {
    local label=$1
    local currentValue=$2
    readInput "$label [${currentValue:-preset/compiler default}] (Enter keeps, - resets): "
    if [[ $promptValue == '-' ]]; then
        promptValue=
    elif [[ -z $promptValue ]]; then
        promptValue=$currentValue
    fi
}

showMainMenu() {
    printf '\nCosmicTiger builder\n'
    printf '  1) Build mode: %s\n' "$buildMode"
    printf '  2) CUDA: %s\n' "$enableCuda"
    printf '  3) Build and run test suite: %s\n' "$testSuite"
    printf '  4) Parallel build jobs: %s\n' "$buildJobs"
    printf '  5) Parallel test jobs: %s\n' "$testJobs"
    printf '  6) HPX installation prefix: %s\n' "${hpxPrefix:-preset default}"
    printf '  7) CUDA architectures: %s\n' "${cudaArchitectures:-${CUDAARCHS:-compiler default}}"
    printf '  Enter or b) Configure and build    q) Quit\n'
    printf 'Build and test jobs share the current local resource budget.\n'
}

dialogChoice() {
    whiptail --clear --backtitle 'CosmicTiger builder' "$@" 3>&1 1>&2 2>&3
}

chooseDialogNumber() {
    local label=$1
    local currentValue=$2
    local numericValue
    while true; do
        if ! promptValue=$(dialogChoice --title "$label" \
            --inputbox "Enter a whole number of jobs from 1 to $coreCount." \
            10 68 "$currentValue"); then
            promptValue=$currentValue
            return
        fi
        if [[ -z $promptValue ]]; then
            promptValue=$currentValue
            return
        fi
        if [[ $promptValue =~ ^[0-9]+$ && ${#promptValue} -le 6 ]]; then
            numericValue=$((10#$promptValue))
            if (( numericValue >= 1 && numericValue <= coreCount )); then
                promptValue=$numericValue
                return
            fi
        fi
        whiptail --clear --backtitle 'CosmicTiger builder' \
            --title 'Invalid job count' \
            --msgbox "Enter a whole number from 1 to $coreCount." 8 58
    done
}

chooseDialogText() {
    local label=$1
    local currentValue=$2
    local description=$3
    if promptValue=$(dialogChoice --title "$label" \
        --inputbox "$description" 10 72 "$currentValue"); then
        if [[ $promptValue == '-' ]]; then
            promptValue=
        fi
    else
        promptValue=$currentValue
    fi
}

if command -v whiptail >/dev/null 2>&1 && [[ ${TERM:-dumb} != dumb ]]; then
    while true; do
        if ! menuChoice=$(dialogChoice --title 'Build configuration' \
            --cancel-button 'Quit' --default-item 1 --notags \
            --menu 'Use arrow keys to choose a setting. Jobs share the current local resource budget.' \
            20 78 9 \
            1 "Build mode: $buildMode" \
            2 "CUDA: $enableCuda" \
            3 "Build and run test suite: $testSuite" \
            4 "Parallel build jobs: $buildJobs" \
            5 "Parallel test jobs: $testJobs" \
            6 "HPX installation prefix: ${hpxPrefix:-preset default}" \
            7 "CUDA architectures: ${cudaArchitectures:-${CUDAARCHS:-compiler default}}" \
            b 'Configure and build' \
            q 'Quit'); then
            printf 'No build started.\n'
            exit 0
        fi
        case "$menuChoice" in
            b) break ;;
            q) printf 'No build started.\n'; exit 0 ;;
            1)
                if promptValue=$(dialogChoice --title 'Build mode' \
                    --default-item "$buildMode" \
                    --menu 'Select a build mode.' 14 64 4 \
                    release 'Optimized, without debug symbols' \
                    relwithdebinfo 'Optimized, with debug symbols' \
                    debug 'Debug checks and symbols' \
                    profile 'Profiling with gprof'); then
                    buildMode=$promptValue
                fi
                ;;
            2|3)
                if [[ $menuChoice == 2 ]]; then
                    dialogLabel='CUDA'
                    dialogCurrent=$enableCuda
                else
                    dialogLabel='Build and run test suite'
                    dialogCurrent=$testSuite
                fi
                if promptValue=$(dialogChoice --title "$dialogLabel" \
                    --default-item "$dialogCurrent" \
                    --menu "Select $dialogLabel." 12 58 2 \
                    ON 'Enabled' OFF 'Disabled'); then
                    if [[ $menuChoice == 2 ]]; then
                        enableCuda=$promptValue
                    else
                        testSuite=$promptValue
                    fi
                fi
                ;;
            4)
                chooseDialogNumber 'Parallel build jobs' "$buildJobs"
                buildJobs=$promptValue
                ;;
            5)
                chooseDialogNumber 'Parallel test jobs' "$testJobs"
                testJobs=$promptValue
                ;;
            6)
                chooseDialogText 'HPX installation prefix' "$hpxPrefix" \
                    'Enter a prefix, or leave blank for the preset default.'
                hpxPrefix=$promptValue
                ;;
            7)
                chooseDialogText 'CUDA architectures' "$cudaArchitectures" \
                    'Enter CUDA architectures, or leave blank for CUDAARCHS/compiler default.'
                cudaArchitectures=$promptValue
                ;;
        esac
    done
else
    while true; do
        showMainMenu
        readInput 'Choice [build]: '
        case "${promptValue,,}" in
            ''|b|build) break ;;
            q|quit) printf 'No build started.\n'; exit 0 ;;
            1) chooseMode ;;
            2) chooseToggle 'Enable CUDA' "$enableCuda"; enableCuda=$promptValue ;;
            3) chooseToggle 'Build and run test suite' "$testSuite"; testSuite=$promptValue ;;
            4) chooseNumber 'Parallel build jobs' "$buildJobs" 1 "$coreCount"; buildJobs=$promptValue ;;
            5) chooseNumber 'Parallel test jobs' "$testJobs" 1 "$coreCount"; testJobs=$promptValue ;;
            6) chooseText 'HPX installation prefix' "$hpxPrefix"; hpxPrefix=$promptValue ;;
            7) chooseText 'CUDA architectures' "$cudaArchitectures"; cudaArchitectures=$promptValue ;;
            *) printf 'Unknown option.\n' ;;
        esac
    done
fi

for requiredCommand in cmake ninja git make gfortran tee mktemp; do
    if ! command -v "$requiredCommand" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$requiredCommand" >&2
        exit 1
    fi
done
if [[ $testSuite == ON ]] && ! command -v ctest >/dev/null 2>&1; then
    printf 'Missing required command: ctest\n' >&2
    exit 1
fi

buildDir="$rootDir/$buildMode"
if ! mkdir -p -- "$buildDir"; then
    printf 'Cannot create build directory: %s\n' "$buildDir" >&2
    exit 1
fi
logFile=$(mktemp "$buildDir/builder.XXXXXX.log") || exit 1

configureCommand=(cmake --preset "$buildMode" -UHPX_DIR '-UMPI_*'
    "-DCOSMICTIGER_ENABLE_CUDA=$enableCuda"
    "-DBUILD_TESTING=$testSuite")
if [[ $testSuite == ON ]]; then
    configureCommand+=("-DCOSMICTIGER_TEST_EIGHT_LOCALITIES=OFF")
fi
if [[ -n $hpxPrefix ]]; then
    if [[ $hpxPrefix == '~/'* && -n ${HOME:-} ]]; then
        hpxPrefix="${HOME}/${hpxPrefix:2}"
    fi
    configureCommand+=("-DCMAKE_PREFIX_PATH=$hpxPrefix")
fi
if [[ $enableCuda == ON ]]; then
    # Clearing the architecture cache does not rerun CUDA detection in an existing build.
    # OFF leaves architecture flags to NVCC when neither the menu nor CUDAARCHS selects them.
    configureCommand+=("-DCMAKE_CUDA_ARCHITECTURES=${cudaArchitectures:-${CUDAARCHS:-OFF}}")
fi
buildCommand=(cmake --build --preset "$buildMode" --target cosmictiger --parallel "$buildJobs")
testBuildCommand=(cmake --build --preset "$buildMode" --target all --parallel "$buildJobs")
testCommand=(ctest --preset "$buildMode" --parallel "$testJobs")

runStage() {
    local stageName=$1
    local commandText
    local -a pipeStatuses
    shift
    printf -v commandText '%q ' "$@"
    printf '\n%s: %s\n' "$stageName" "$commandText"
    printf '\n%s: %s\n' "$stageName" "$commandText" >> "$logFile" || return 1
    "$@" 2>&1 | tee -a "$logFile"
    pipeStatuses=("${PIPESTATUS[@]}")
    if (( pipeStatuses[1] != 0 )); then
        printf '%s could not write the log: %s\n' "$stageName" "$logFile" >&2
        return 1
    fi
    if (( pipeStatuses[0] != 0 )); then
        printf '%s failed (exit %s). Log: %s\n' "$stageName" "${pipeStatuses[0]}" "$logFile" >&2
        return "${pipeStatuses[0]}"
    fi
    printf '%s succeeded.\n' "$stageName"
}

printf '\nBuild directory: %s\nLog: %s\n' "$buildDir" "$logFile"
runStage 'Initialize CAMB submodules' git submodule update --init --recursive -- third_party/camb || exit $?
runStage Configure "${configureCommand[@]}" || exit $?
runStage 'Build cosmictiger' "${buildCommand[@]}" || exit $?
if [[ $testSuite == ON ]]; then
    runStage 'Build test suite' "${testBuildCommand[@]}" || exit $?
    runStage 'Run test suite' "${testCommand[@]}" || exit $?
fi
printf '\nCosmicTiger build succeeded: %s\nLog: %s\n' "$buildDir" "$logFile"
if [[ $testSuite == ON ]]; then
    printf 'Test suite passed.\n'
else
    printf 'Test suite was not requested.\n'
fi
