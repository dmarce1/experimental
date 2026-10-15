/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <cosmictiger/camb.hpp>
#include <cosmictiger/debug.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <hpx/components/process/util/execute.hpp>
#include <hpx/components/process/util/initializers.hpp>
#include <hpx/components/process/util/posix/initializers/bind_fd.hpp>
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>
#include <hpx/components/process/util/wait_for_exit.hpp>
#include <hpx/runtime_local/run_as_os_thread.hpp>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace cosmictiger {
namespace detail {

    int cambComponent(MatterComponent component) {
        PROFILE();
        switch (component) {
        case MatterComponent::totalMatter:
            return 7;
        case MatterComponent::cdmAndBaryons:
            return 8;
        case MatterComponent::cdm:
            return 2;
        }
        expect(false);
        return 0;
    }

    class CambRunDirectory {
    public:
        CambRunDirectory() {
            PROFILE();
            std::string pattern = (hpx::filesystem::temp_directory_path() / "cosmictiger-camb-XXXXXX").string();
            char* created = ::mkdtemp(pattern.data());
            if (created == nullptr) {
                throw std::system_error(errno, std::generic_category(), "Cannot create CAMB run directory");
            }
            path_ = created;
        }

        CambRunDirectory(CambRunDirectory const&) = delete;
        CambRunDirectory& operator=(CambRunDirectory const&) = delete;

        ~CambRunDirectory() {
            PROFILE();
            if (remove_) {
                std::error_code error;
                hpx::filesystem::remove_all(path_, error);
            }
        }

        hpx::filesystem::path const& path() const noexcept {
            PROFILE();
            return path_;
        }

        void removeAfterSuccess() noexcept {
            PROFILE();
            remove_ = true;
        }

    private:
        hpx::filesystem::path path_;
        bool remove_ = false;
    };

    class BorrowedFileDescriptor {
    public:
        explicit BorrowedFileDescriptor(int descriptor)
          : descriptor_(descriptor) {
            PROFILE();
        }

        int handle() const noexcept {
            PROFILE();
            return descriptor_;
        }

    private:
        int descriptor_;
    };

    class ParentForkError : public hpx::components::process::posix::initializers::initializer_base {
    public:
        template <typename Executor>
        void on_fork_error(Executor&) const {
            PROFILE();
            // Preserve fork's errno before profiling the parent-only failure path.
            int const code = errno;
            auto reportFailure = hpx::annotated_function(
                [code] {
                    PROFILE();
                    throw std::system_error(code, std::generic_category(), "Cannot start CAMB process");
                },
                "camb.forkError");
            reportFailure();
        }
    };

    void writeCambInput(hpx::filesystem::path const& path, CosmologyParameters const& p, MatterPowerRequest const& r) {
        PROFILE();
        std::ofstream file(path);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.imbue(std::locale::classic());
        file << std::setprecision(std::numeric_limits<double>::max_digits10);
        FpeGuard guard;
        double const neutrinoCount = static_cast<double>(expectNonNegative(p.massiveNeutrinos)) + expectNonNegative(p.masslessNeutrinos);
        double const effectiveNeutrinos = expectNonNegative(p.effectiveNeutrinos);
        expectEqual(neutrinoCount == 0.0, effectiveNeutrinos == 0.0);
        double const neutrinoDegeneracy = neutrinoCount == 0.0 ? 0.0 : effectiveNeutrinos / neutrinoCount;
        file << "output_root = result\n"
             << "get_scalar_cls = F\nget_vector_cls = F\nget_tensor_cls = F\nget_transfer = T\n"
             << "do_lensing = F\ndo_nonlinear = 0\ninitial_condition = 1\n"
             << "use_physical = T\nhubble = " << expectPositive(p.hubble) << '\n'
             << "ombh2 = " << expectPositive(p.omegaBaryonH2) << '\n'
             << "omch2 = " << expectNonNegative(p.omegaCdmH2) << '\n'
             << "omnuh2 = " << expectNonNegative(p.omegaNeutrinoH2) << '\n';
        expectEqual(p.omegaNeutrinoH2 > 0.0, p.massiveNeutrinos > 0);
        file << "omk = " << p.omegaCurvature << '\n'
             << "massless_neutrinos = " << p.masslessNeutrinos * neutrinoDegeneracy << '\n'
             << "nu_mass_eigenstates = 1\nmassive_neutrinos = " << p.massiveNeutrinos << '\n'
             << "share_delta_neff = F\nnu_mass_degeneracies = " << p.massiveNeutrinos * neutrinoDegeneracy << '\n'
             << "nu_mass_fractions = 1\n"
             << "initial_power_num = 1\nscalar_amp(1) = " << expectPositive(p.scalarAmplitude) << '\n'
             << "scalar_spectral_index(1) = " << p.spectralIndex << '\n'
             << "scalar_nrun(1) = " << p.spectralRunning << '\n'
             << "scalar_nrunrun(1) = 0\npivot_scalar = " << expectPositive(p.pivotMpcInverse) << '\n';
        expectNonPositive(p.darkEnergyW + p.darkEnergyWa);
        file << "dark_energy_model = PPF\nw = " << p.darkEnergyW << '\n'
             << "wa = " << p.darkEnergyWa << "\ncs2_lam = 1\n"
             << "temp_cmb = " << expectPositive(p.temperatureCmb) << '\n';
        expect(p.heliumFraction < 1.0);
        file << "helium_fraction = " << expectNonNegative(p.heliumFraction) << '\n'
             << "reionization = T\nre_use_optical_depth = T\nre_optical_depth = " << expectNonNegative(p.opticalDepth) << '\n'
             << "recombination_model = Recfast\n"
             << "transfer_high_precision = T\naccurate_massive_neutrino_transfers = T\n"
             // The Fortran INI uses h/Mpc, whereas Python set_matter_power uses Mpc^-1.
             << "transfer_kmax = " << expectPositive(r.maximumWaveNumber) << '\n'
             << "transfer_k_per_logint = " << expectNonNegative(r.samplesPerLogInterval) << '\n'
             << "transfer_num_redshifts = 1\ntransfer_redshift(1) = " << expectNonNegative(r.redshift) << '\n'
             << "transfer_interp_matterpower = F\ntransfer_filename(1) = transfer.dat\n"
             << "transfer_matterpower(1) = matterpower.dat\ntransfer_power_var = " << cambComponent(r.component) << '\n'
             << "accuracy_boost = " << expectPositive(r.accuracyBoost) << '\n'
             << "number_of_threads = 1\nfeedback_level = 0\noutput_file_headers = T\n"
             << "derived_parameters = F\nprint_fortran_warnings = T\n";
        file.close();
    }

    int runCamb(hpx::filesystem::path const& executable, hpx::filesystem::path const& directory) {
        PROFILE();
        auto const logPath = directory / "camb.log";
        auto closeLog = hpx::annotated_function(
            [](std::FILE* file) {
                PROFILE();
                std::fclose(file);
            },
            "camb.closeLog");
        std::unique_ptr<std::FILE, decltype(closeLog)> log(std::fopen(logPath.c_str(), "w"), closeLog);
        if (!log) {
            throw std::system_error(errno, std::generic_category(), "Cannot open CAMB log");
        }
        int const descriptor = ::fileno(log.get());
        if (::fcntl(descriptor, F_SETFD, FD_CLOEXEC) == -1) {
            throw std::system_error(errno, std::generic_category(), "Cannot set CAMB log descriptor flags");
        }
        namespace process = hpx::components::process;
        std::vector<std::string> const arguments{executable.string(), "parameters.ini"};
        std::vector<std::string> const environment{"OMP_NUM_THREADS=1", "OMP_THREAD_LIMIT=1", "OMP_DYNAMIC=FALSE", "OPENBLAS_NUM_THREADS=1", "LC_ALL=C"};
        BorrowedFileDescriptor const borrowed(descriptor);
        auto const child = process::util::execute(process::run_exe(executable.string()), process::set_args(arguments), process::set_env(environment),
            process::start_in_dir(directory.string()), process::bind_fd(STDOUT_FILENO, borrowed), process::bind_fd(STDERR_FILENO, borrowed),
            process::close_stdin(), ParentForkError{});
        // Failed execs leave through HPX's _exit(EXIT_FAILURE) path. The parent
        // reaps the child below; no project callback invokes HPX after fork.
        return process::util::wait_for_exit(child);
    }

    std::vector<MatterPowerSample> readCambSpectrum(hpx::filesystem::path const& path) {
        PROFILE();
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("CAMB did not produce a matter power spectrum");
        }
        file.imbue(std::locale::classic());
        std::vector<MatterPowerSample> samples;
        std::string line;
        while (std::getline(file, line)) {
            auto const content = line.find_first_not_of(" \t\r");
            if (content == std::string::npos || line[content] == '#') {
                continue;
            }
            // Accept both Fortran exponent spellings without locale-dependent parsing.
            for (char& character : line) {
                if (character == 'D' || character == 'd') {
                    character = 'E';
                }
            }
            std::istringstream row(line);
            row.imbue(std::locale::classic());
            MatterPowerSample sample{};
            std::string extra;
            if (!(row >> sample.waveNumber >> sample.power) || (row >> extra)) {
                throw std::runtime_error("Malformed row in CAMB matter power spectrum");
            }
            if (!(sample.waveNumber > 0.0) || !(sample.power > 0.0) || (!samples.empty() && !(sample.waveNumber > samples.back().waveNumber))) {
                throw std::runtime_error("CAMB matter power rows require increasing positive k and positive P(k)");
            }
            samples.push_back(sample);
        }
        if (file.bad()) {
            throw std::runtime_error("Cannot read CAMB matter power spectrum");
        }
        if (samples.size() < 2) {
            throw std::runtime_error("CAMB matter power spectrum contains fewer than two rows");
        }
        return samples;
    }

    MatterPowerSpectrum computeCamb(hpx::filesystem::path const& executable, CosmologyParameters const& p, MatterPowerRequest const& r) {
        PROFILE();
        CambRunDirectory run;
        try {
            writeCambInput(run.path() / "parameters.ini", p, r);
            int const status = runCamb(executable, run.path());
            if (status != 0) {
                throw std::runtime_error("CAMB exited with status " + std::to_string(status));
            }
            MatterPowerSpectrum result(r.redshift, p.hubble / 100.0, r.component, readCambSpectrum(run.path() / "result_matterpower.dat"));
            // A successful exit alone does not establish that the requested k range was computed.
            // CAMB's text output rounds k to about seven significant figures.
            if (result.samples().back().waveNumber < r.maximumWaveNumber * (1.0 - 2.0e-6)) {
                throw std::runtime_error("CAMB spectrum does not cover the requested maximum wave number");
            }
            run.removeAfterSuccess();
            return result;
        } catch (std::exception const& error) {
            throw std::runtime_error(std::string(error.what()) + "; CAMB input, output and log retained in " + run.path().string());
        }
    }

}    // namespace detail

CambSolver::CambSolver(hpx::filesystem::path executable) {
    PROFILE();
    expect(!executable.empty());
    executable_ = hpx::filesystem::absolute(std::move(executable));
}

hpx::filesystem::path CambSolver::defaultExecutable() {
    PROFILE();
    return COSMICTIGER_CAMB_EXECUTABLE;
}

hpx::filesystem::path const& CambSolver::executable() const noexcept {
    PROFILE();
    return executable_;
}

MatterPowerSpectrum CambSolver::compute(CosmologyParameters const& parameters, MatterPowerRequest const& request) const {
    PROFILE();
    return hpx::run_as_os_thread(hpx::annotated_function(
                                     [path = executable_, parameters, request]() {
                                         PROFILE();
                                         return detail::computeCamb(path, parameters, request);
                                     },
                                     "camb.computeOnOsThread"))
        .get();
}

}    // namespace cosmictiger
