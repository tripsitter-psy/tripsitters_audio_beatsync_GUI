#pragma once

// FORWARDER HEADER: Redirects to the canonical FBeatsyncLoader declaration.
// The authoritative header is in the TripSitterUE plugin:
//   Plugins/TripSitterUE/TripSitterUE/Public/BeatsyncLoader.h
//
// This uses a relative path to avoid self-inclusion ambiguity when both
// this module and the plugin have a BeatsyncLoader.h in their Public folders.
#include "../../../Plugins/TripSitterUE/TripSitterUE/Public/BeatsyncLoader.h"
