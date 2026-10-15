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

#include <cosmictiger/debug.hpp>
#include <cosmictiger/einstein_boltzmann.hpp>
#include <cosmictiger/profiler.hpp>

#include <cmath>
#include <stddef.h>
#include <utility>

namespace cosmictiger {
namespace detail {

    bool validComponent(MatterComponent component) {
        PROFILE();
        switch (component) {
        case MatterComponent::totalMatter:
        case MatterComponent::cdmAndBaryons:
        case MatterComponent::cdm:
            return true;
        }
        return false;
    }

}    // namespace detail

MatterPowerSpectrum::MatterPowerSpectrum(double redshift, double hubbleParameter, MatterComponent component, std::vector<MatterPowerSample> samples)
  : redshift_(expectNonNegative(redshift))
  , hubbleParameter_(expectPositive(hubbleParameter))
  , component_(component)
  , samples_(std::move(samples)) {
    PROFILE();
    expect(detail::validComponent(component_));
    expect(samples_.size() >= 2);
}

double MatterPowerSpectrum::redshift() const noexcept {
    PROFILE();
    return redshift_;
}

double MatterPowerSpectrum::hubbleParameter() const noexcept {
    PROFILE();
    return hubbleParameter_;
}

MatterComponent MatterPowerSpectrum::component() const noexcept {
    PROFILE();
    return component_;
}

std::vector<MatterPowerSample> const& MatterPowerSpectrum::samples() const noexcept {
    PROFILE();
    return samples_;
}

double MatterPowerSpectrum::power(double waveNumber) const {
    PROFILE();
    FpeGuard guard;
    waveNumber = expectRange(expectPositive(samples_.front().waveNumber), expectPositive(waveNumber), expectPositive(samples_.back().waveNumber));
    size_t first = 0;
    size_t last = samples_.size() - 1;
    while (first < last) {
        size_t const middle = first + (last - first) / 2;
        expectPositive(samples_[middle].waveNumber);
        if (middle > 0) {
            expect(samples_[middle].waveNumber > samples_[middle - 1].waveNumber);
        }
        if (middle + 1 < samples_.size()) {
            expect(samples_[middle].waveNumber < samples_[middle + 1].waveNumber);
        }
        if (samples_[middle].waveNumber < waveNumber) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    if (samples_[first].waveNumber == waveNumber) {
        return expectPositive(samples_[first].power);
    }
    auto const& lower = samples_[expectPositive(first) - 1];
    auto const& upper = samples_[first];
    double const lowerLogWaveNumber = std::log(expectPositive(lower.waveNumber));
    double const upperLogWaveNumber = std::log(expectPositive(upper.waveNumber));
    double const weight = (std::log(waveNumber) - lowerLogWaveNumber) / expectPositive(upperLogWaveNumber - lowerLogWaveNumber);
    return std::exp((1.0 - weight) * std::log(expectPositive(lower.power)) + weight * std::log(expectPositive(upper.power)));
}

}    // namespace cosmictiger
