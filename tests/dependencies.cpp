#include <gtest/gtest.h>
#include <mpi.h>
#include <silo.h>

TEST(Dependencies, SiloLibraryIsAvailable) {
    const char* version = DBVersion();
    ASSERT_NE(version, nullptr);
    EXPECT_NE(version[0], '\0');
}

TEST(Dependencies, MpiLibraryIsAvailable) {
    int version = 0;
    int subversion = 0;
    ASSERT_EQ(MPI_Get_version(&version, &subversion), MPI_SUCCESS);
    EXPECT_EQ(version, MPI_VERSION);
    EXPECT_EQ(subversion, MPI_SUBVERSION);
}
