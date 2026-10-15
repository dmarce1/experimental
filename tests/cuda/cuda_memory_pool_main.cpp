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

#include <cosmictiger/profiler.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>

int runGpuTests();

void yieldForGpuSubmission() {
    PROFILE();
    hpx::this_thread::yield();
}

int hpx_main() {
    PROFILE();
    int const result = runGpuTests();
    hpx::finalize();
    return result;
}

int main(int argc, char* argv[]) {
    PROFILE();
    hpx::init_params params;
    params.cfg = {"hpx.os_threads=2"};
    return hpx::init(argc, argv, params);
}
