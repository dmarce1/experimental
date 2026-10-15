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
#include <gtest/gtest.h>
#include <mpi.h>
#include <silo.h>

TEST(Dependencies, SiloLibraryIsAvailable) {
    PROFILE();
    char const* version = DBVersion();
    ASSERT_NE(version, nullptr);
    EXPECT_NE(version[0], '\0');
}

TEST(Dependencies, MpiLibraryIsAvailable) {
    PROFILE();
    int version = 0;
    int subversion = 0;
    ASSERT_EQ(MPI_Get_version(&version, &subversion), MPI_SUCCESS);
    EXPECT_EQ(version, MPI_VERSION);
    EXPECT_EQ(subversion, MPI_SUBVERSION);
}
