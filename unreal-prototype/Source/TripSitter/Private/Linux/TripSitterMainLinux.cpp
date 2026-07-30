// TripSitter - Linux Entry Point

#include "CoreMinimal.h"
#include "UnixCommonStartup.h"

// Forward declaration
int RunTripSitter(const TCHAR* CommandLine);

int main(int argc, char* argv[])
{
    return CommonUnixMain(argc, argv, &RunTripSitter);
}
