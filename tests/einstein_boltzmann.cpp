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
#include <cosmictiger/einstein_boltzmann.hpp>
#include <cosmictiger/profiler.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <gtest/gtest.h>
#include <hpx/include/async.hpp>
#include <hpx/modules/filesystem.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using cosmictiger::CambSolver;
using cosmictiger::CosmologyParameters;
using cosmictiger::EinsteinBoltzmannSolver;
using cosmictiger::MatterComponent;
using cosmictiger::MatterPowerRequest;
using cosmictiger::MatterPowerSample;
using cosmictiger::MatterPowerSpectrum;

class AnalyticSolver final : public EinsteinBoltzmannSolver {
public:
    MatterPowerSpectrum compute(CosmologyParameters const& parameters, MatterPowerRequest const& request) const override {
        PROFILE();
        double const growth = 1.0 / (1.0 + request.redshift);
        double const amplitude = parameters.scalarAmplitude * growth * growth;
        return {request.redshift, parameters.hubble / 100.0, request.component, {{0.01, amplitude}, {0.1, 100.0 * amplitude}}};
    }
};

class TestDirectory {
public:
    TestDirectory() {
        PROFILE();
        std::string directoryTemplate = (hpx::filesystem::temp_directory_path() / "cosmictiger_boltzmann_test_XXXXXX").string();
        if (::mkdtemp(directoryTemplate.data()) == nullptr) {
            throw std::runtime_error("could not create Einstein-Boltzmann test directory");
        }
        directory_ = directoryTemplate;
    }

    TestDirectory(TestDirectory const&) = delete;
    TestDirectory& operator=(TestDirectory const&) = delete;

    ~TestDirectory() {
        PROFILE();
        std::error_code error;
        hpx::filesystem::remove_all(directory_, error);
    }

    hpx::filesystem::path file(std::string const& name) const {
        PROFILE();
        return directory_ / name;
    }

    hpx::filesystem::path executable(std::string const& body) const {
        PROFILE();
        auto const path = file("fake_camb.sh");
        std::ofstream output(path);
        output << "#!/bin/sh\n" << body;
        output.close();
        if (!output) {
            throw std::runtime_error("could not write fake CAMB executable");
        }
        hpx::filesystem::permissions(path, hpx::filesystem::perms::owner_read | hpx::filesystem::perms::owner_write | hpx::filesystem::perms::owner_exec);
        return path;
    }

private:
    hpx::filesystem::path directory_;
};

MatterPowerSpectrum computeOnTask(EinsteinBoltzmannSolver const& solver, CosmologyParameters const& parameters, MatterPowerRequest const& request) {
    PROFILE();
    return hpx::async(hpx::launch::async,
        hpx::annotated_function(
            [&] {
                PROFILE();
                return solver.compute(parameters, request);
            },
            "test.camb.computeSpectrum"))
        .get();
}

void expectFailedRun(CambSolver const& solver, char const* diagnostic, MatterPowerRequest const& request = {},
    hpx::function<void(hpx::filesystem::path const&)> inspectRun = {}) {
    PROFILE();
    try {
        (void) computeOnTask(solver, {}, request);
        FAIL() << "CAMB adapter accepted a failed run";
    } catch (std::runtime_error const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find(diagnostic), std::string::npos) << message;
        std::string const marker = "; CAMB input, output and log retained in ";
        auto const position = message.rfind(marker);
        ASSERT_NE(position, std::string::npos) << message;
        hpx::filesystem::path const directory(message.substr(position + marker.size()));
        // Only clean the private CAMB run created by this test.
        ASSERT_EQ(directory.parent_path(), hpx::filesystem::temp_directory_path());
        ASSERT_EQ(directory.filename().string().find("cosmictiger-camb-"), 0U);
        EXPECT_TRUE(hpx::filesystem::is_regular_file(directory / "parameters.ini"));
        EXPECT_TRUE(hpx::filesystem::is_regular_file(directory / "camb.log"));
        if (inspectRun) {
            inspectRun(directory);
        }
        hpx::filesystem::remove_all(directory);
    }
}

TEST(EinsteinBoltzmann, AlternateSolverUsesCommonContract) {
    PROFILE();
    AnalyticSolver const implementation;
    EinsteinBoltzmannSolver const& solver = implementation;
    CosmologyParameters parameters;
    parameters.scalarAmplitude = 8.0e-9;
    parameters.hubble = 72.0;
    MatterPowerRequest request;
    request.redshift = 1.0;
    request.component = MatterComponent::cdm;

    auto const spectrum = computeOnTask(solver, parameters, request);
    EXPECT_EQ(spectrum.redshift(), request.redshift);
    EXPECT_DOUBLE_EQ(spectrum.hubbleParameter(), 0.72);
    EXPECT_EQ(spectrum.component(), request.component);
    ASSERT_EQ(spectrum.samples().size(), 2U);
    EXPECT_NEAR(spectrum.power(std::sqrt(0.001)), 2.0e-8, 1.0e-20);
}

TEST(MatterPowerSpectrum, InterpolatesPowerLawAndReturnsExactEndpoints) {
    PROFILE();
    MatterPowerSpectrum const spectrum(49.0, 0.674, MatterComponent::totalMatter, {{0.01, 3.0}, {0.1, 300.0}, {1.0, 30000.0}});
    EXPECT_DOUBLE_EQ(spectrum.power(0.01), 3.0);
    EXPECT_DOUBLE_EQ(spectrum.power(0.1), 300.0);
    EXPECT_DOUBLE_EQ(spectrum.power(1.0), 30000.0);
    EXPECT_NEAR(spectrum.power(0.03), 27.0, 1.0e-12);
    EXPECT_NEAR(spectrum.power(0.3), 2700.0, 1.0e-10);
}

TEST(MatterPowerSpectrum, RefusesExtrapolationAndInvalidQueries) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    MatterPowerSpectrum const spectrum(0.0, 0.7, MatterComponent::cdm, {{0.01, 1.0}, {0.1, 2.0}});
    EXPECT_THROW(spectrum.power(0.009), std::runtime_error);
    EXPECT_THROW(spectrum.power(0.101), std::runtime_error);
    EXPECT_THROW(spectrum.power(0.0), std::runtime_error);
    EXPECT_THROW(spectrum.power(-0.1), std::runtime_error);
}

TEST(MatterPowerSpectrum, RejectsIncompleteTablesAndUnorderedSamplesAsConsumed) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    EXPECT_THROW(MatterPowerSpectrum(0.0, 0.7, MatterComponent::totalMatter, {}), std::runtime_error);
    EXPECT_THROW(MatterPowerSpectrum(0.0, 0.7, MatterComponent::totalMatter, {{0.1, 1.0}}), std::runtime_error);
    auto check = [](std::vector<MatterPowerSample> samples, double query) {
        PROFILE();
        MatterPowerSpectrum const spectrum(0.0, 0.7, MatterComponent::totalMatter, std::move(samples));
        EXPECT_THROW(spectrum.power(query), std::runtime_error);
    };
    check({{0.1, 1.0}, {0.1, 2.0}}, 0.1);
    check({{0.2, 1.0}, {0.1, 2.0}}, 0.15);
    check({{0.01, 1.0}, {0.1, 2.0}, {0.05, 3.0}}, 0.03);
}

TEST(MatterPowerSpectrum, RejectsNonpositiveSamplesAsConsumed) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    for (double const badValue : {0.0, -1.0}) {
        MatterPowerSpectrum const badWaveNumber(0.0, 0.7, MatterComponent::totalMatter, {{badValue, 1.0}, {0.1, 2.0}});
        EXPECT_THROW(badWaveNumber.power(0.05), std::runtime_error);
        MatterPowerSpectrum const badPower(0.0, 0.7, MatterComponent::totalMatter, {{0.01, badValue}, {0.1, 2.0}});
        EXPECT_THROW(badPower.power(0.01), std::runtime_error);
        EXPECT_THROW(badPower.power(0.05), std::runtime_error);
    }
}

TEST(MatterPowerSpectrum, RejectsInvalidMetadata) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    std::vector<MatterPowerSample> const samples{{0.01, 1.0}, {0.1, 2.0}};
    EXPECT_THROW(MatterPowerSpectrum(-1.0, 0.7, MatterComponent::totalMatter, samples), std::runtime_error);
    EXPECT_THROW(MatterPowerSpectrum(0.0, 0.0, MatterComponent::totalMatter, samples), std::runtime_error);
    EXPECT_THROW(MatterPowerSpectrum(0.0, 0.7, static_cast<MatterComponent>(-1), samples), std::runtime_error);
}

void expectInvalidCambInput(CambSolver const& solver, CosmologyParameters const& parameters, MatterPowerRequest const& request) {
    PROFILE();
    try {
        (void) computeOnTask(solver, parameters, request);
        FAIL() << "CAMB adapter accepted an invalid input";
    } catch (std::runtime_error const& error) {
        std::string const message = error.what();
        EXPECT_NE(message.find("CosmicTiger expectation failed"), std::string::npos) << message;
        std::string const marker = "; CAMB input, output and log retained in ";
        auto const position = message.rfind(marker);
        ASSERT_NE(position, std::string::npos) << message;
        hpx::filesystem::path const directory(message.substr(position + marker.size()));
        ASSERT_EQ(directory.parent_path(), hpx::filesystem::temp_directory_path());
        ASSERT_EQ(directory.filename().string().find("cosmictiger-camb-"), 0U);
        EXPECT_FALSE(hpx::filesystem::exists(directory / "camb.log"));
        hpx::filesystem::remove_all(directory);
    }
}

TEST(CambSolver, RejectsInvalidCosmologyWhileWritingInputBeforeStartingProcess) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    CambSolver const solver("/bin/true");
    MatterPowerRequest const request;
    auto check = [&](auto change) {
        PROFILE();
        CosmologyParameters parameters;
        change(parameters);
        expectInvalidCambInput(solver, parameters, request);
    };
    check([](auto& parameters) {
        PROFILE();
        parameters.hubble = 0.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.omegaBaryonH2 = -0.1;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.scalarAmplitude = 0.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.pivotMpcInverse = 0.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.omegaNeutrinoH2 = -0.1;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.omegaNeutrinoH2 = 0.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.massiveNeutrinos = 0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.massiveNeutrinos = -1;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.masslessNeutrinos = -1;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.effectiveNeutrinos = -1.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.omegaNeutrinoH2 = 0.0;
        parameters.massiveNeutrinos = 0;
        parameters.masslessNeutrinos = 0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.heliumFraction = 1.0;
    });
    check([](auto& parameters) {
        PROFILE();
        parameters.darkEnergyW = -0.5;
        parameters.darkEnergyWa = 0.6;
    });
}

TEST(CambSolver, RejectsInvalidMatterPowerRequestsWhileWritingInputBeforeStartingProcess) {
    PROFILE();
    if (!cosmictiger::validationChecksEnabled()) {
        GTEST_SKIP() << "Validation checks are disabled";
    }
    CambSolver const solver("/bin/true");
    CosmologyParameters const parameters;
    auto check = [&](auto change) {
        PROFILE();
        MatterPowerRequest request;
        change(request);
        expectInvalidCambInput(solver, parameters, request);
    };
    check([](auto& request) {
        PROFILE();
        request.redshift = -1.0;
    });
    check([](auto& request) {
        PROFILE();
        request.maximumWaveNumber = 0.0;
    });
    check([](auto& request) {
        PROFILE();
        request.samplesPerLogInterval = -1;
    });
    check([](auto& request) {
        PROFILE();
        request.accuracyBoost = 0.0;
    });
    check([](auto& request) {
        PROFILE();
        request.component = static_cast<MatterComponent>(-1);
    });
}

TEST(CambSolver, ReportsMissingExecutable) {
    PROFILE();
    TestDirectory const temporary;
    expectFailedRun(CambSolver(temporary.file("missing_camb")), "CAMB");
}

TEST(CambSolver, ReportsFailedExternalProcess) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable("printf 'synthetic CAMB failure\\n'\nexit 23\n"));
    expectFailedRun(solver, "exited with status");
}

TEST(CambSolver, RejectsSuccessfulProcessWithoutSpectrum) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable("exit 0\n"));
    expectFailedRun(solver, "did not produce a matter power spectrum");
}

TEST(CambSolver, PreservesRequestedNeutrinoCountsAndTotalEffectiveNumber) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable("exit 23\n"));
    expectFailedRun(solver, "exited with status", {},
        hpx::annotated_function(
            [](hpx::filesystem::path const& directory) {
                PROFILE();
                std::ifstream input(directory / "parameters.ini");
                double masslessEffective = std::numeric_limits<double>::quiet_NaN();
                double massiveEffective = std::numeric_limits<double>::quiet_NaN();
                int massiveCount = -1;
                std::string shareDeltaNeff;
                std::string line;
                while (std::getline(input, line)) {
                    std::istringstream row(line);
                    std::string key;
                    std::string separator;
                    std::string value;
                    if (!(row >> key >> separator >> value)) {
                        continue;
                    }
                    if (key == "massless_neutrinos") {
                        masslessEffective = std::stod(value);
                    } else {
                        if (key == "nu_mass_degeneracies") {
                            massiveEffective = std::stod(value);
                        } else {
                            if (key == "massive_neutrinos") {
                                massiveCount = std::stoi(value);
                            } else {
                                if (key == "share_delta_neff") {
                                    shareDeltaNeff = value;
                                }
                            }
                        }
                    }
                }
                EXPECT_EQ(massiveCount, 1);
                EXPECT_EQ(shareDeltaNeff, "F");
                EXPECT_NEAR(masslessEffective, 2.0306666666666664, 1.0e-14);
                EXPECT_NEAR(massiveEffective, 1.0153333333333332, 1.0e-14);
                EXPECT_NEAR(masslessEffective + massiveEffective, 3.046, 1.0e-14);
            },
            "test.camb.inspectFailedRun"));
}

TEST(CambSolver, RejectsMalformedExternalSpectrum) {
    PROFILE();
    TestDirectory const temporary;
    // Read the requested path from the input rather than depend on a particular
    // temporary directory name or output-root choice in the CAMB adapter.
    CambSolver const solver(temporary.executable(R"(outputRoot=
powerFile=
while IFS='=' read -r key value; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$key" in
        *output_root*) outputRoot="$value" ;;
        *transfer_matterpower\(1\)*) powerFile="$value" ;;
    esac
done < "$1"
if [ -n "$outputRoot" ]; then
    powerFile="${outputRoot}_${powerFile}"
fi
printf '# k/h P\n0.01 1.0\n0.02 invalid\n' > "$powerFile"
exit 0
)"));
    expectFailedRun(solver, "Malformed row");
}

TEST(CambSolver, RejectsUnorderedExternalSpectrum) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable("printf '0.02 1.0\\n0.01 2.0\\n' > result_matterpower.dat\nexit 0\n"));
    expectFailedRun(solver, "increasing positive k");
}

TEST(CambSolver, RejectsSpectrumShortOfRequestedRange) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable("printf '0.01 1.0\\n0.1 2.0\\n' > result_matterpower.dat\nexit 0\n"));
    MatterPowerRequest request;
    request.maximumWaveNumber = 0.5;
    expectFailedRun(solver, "does not cover the requested maximum", request);
}

TEST(CambSolver, ReadsFortranExponentAndRemovesSuccessfulRunFiles) {
    PROFILE();
    TestDirectory const temporary;
    CambSolver const solver(temporary.executable(R"(printf '%s\n' "$PWD" > "$0.run_directory"
printf '# k/h P\n0.01 1.0D+2\n0.5 4.0d+2\n' > result_matterpower.dat
exit 0
)"));
    MatterPowerRequest request;
    request.maximumWaveNumber = 0.5;
    CosmologyParameters parameters;
    parameters.omegaNeutrinoH2 = 0.0;
    parameters.massiveNeutrinos = 0;
    parameters.masslessNeutrinos = 0;
    parameters.effectiveNeutrinos = 0.0;
    auto const spectrum = computeOnTask(solver, parameters, request);
    EXPECT_DOUBLE_EQ(spectrum.power(0.01), 100.0);
    EXPECT_DOUBLE_EQ(spectrum.power(0.5), 400.0);
    std::ifstream record(temporary.file("fake_camb.sh.run_directory"));
    std::string runDirectory;
    ASSERT_TRUE(std::getline(record, runDirectory));
    EXPECT_FALSE(hpx::filesystem::exists(runDirectory));
}

TEST(CambSolver, LinearMatterSpectrumGrowthNormalizationAndComponents) {
    PROFILE();
    CambSolver const solver;
    CosmologyParameters parameters;
    MatterPowerRequest request;
    request.redshift = 0.0;
    request.maximumWaveNumber = 0.5;
    request.samplesPerLogInterval = 20;

    auto const present = computeOnTask(solver, parameters, request);
    EXPECT_EQ(present.redshift(), 0.0);
    EXPECT_DOUBLE_EQ(present.hubbleParameter(), parameters.hubble / 100.0);
    EXPECT_EQ(present.component(), MatterComponent::totalMatter);
    ASSERT_GT(present.samples().size(), 20U);
    EXPECT_LT(present.samples().front().waveNumber, 0.01);
    EXPECT_GE(present.samples().back().waveNumber, request.maximumWaveNumber);
    double const presentPower = present.power(0.1);
    // A broad independent LCDM plausibility check also catches unit mistakes.
    EXPECT_GT(presentPower, 3500.0);
    EXPECT_LT(presentPower, 7500.0);

    request.redshift = 49.0;
    auto const initial = computeOnTask(solver, parameters, request);
    EXPECT_EQ(initial.redshift(), 49.0);
    double const growthRatio = initial.power(0.1) / presentPower;
    EXPECT_GT(growthRatio, 1.0e-4);
    EXPECT_LT(growthRatio, 1.0e-2);

    request.redshift = 0.0;
    parameters.scalarAmplitude *= 2.0;
    auto const doubled = computeOnTask(solver, parameters, request);
    for (double const waveNumber : {0.01, 0.1, 0.4}) {
        EXPECT_NEAR(doubled.power(waveNumber) / present.power(waveNumber), 2.0, 2.0e-5);
    }

    parameters = CosmologyParameters{};
    parameters.omegaNeutrinoH2 = 0.005;
    parameters.massiveNeutrinos = 3;
    parameters.masslessNeutrinos = 0;
    auto const total = computeOnTask(solver, parameters, request);
    request.component = MatterComponent::cdmAndBaryons;
    auto const clustered = computeOnTask(solver, parameters, request);
    EXPECT_EQ(clustered.component(), MatterComponent::cdmAndBaryons);
    // Freestreaming neutrinos reduce the total density contrast relative to
    // the clustering CDM+baryon contrast at the same cosmology and redshift.
    double const componentRatio = total.power(0.4) / clustered.power(0.4);
    EXPECT_GT(componentRatio, 0.8);
    EXPECT_LT(componentRatio, 0.999);
}

}    // namespace
