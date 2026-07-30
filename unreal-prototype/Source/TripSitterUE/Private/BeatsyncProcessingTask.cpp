// BeatsyncProcessingTask.cpp - Source/TripSitterUE module
//
// NOTE: This file is intentionally empty. The BeatsyncProcessingTask implementation
// is provided by the TripSitterUE plugin (Plugins/TripSitterUE/TripSitterUE/Private/).
//
// The previous approach of #include-ing the TripSitter/Private/BeatsyncProcessingTask.cpp
// caused ODR (One Definition Rule) violations when multiple translation units were compiled.
//
// Build configuration should ensure:
// - TripSitter Program target uses its own BeatsyncProcessingTask.cpp in Source/TripSitter/Private/
// - TripSitterUE plugin uses its own BeatsyncProcessingTask.cpp in Plugins/TripSitterUE/TripSitterUE/Private/
// - This module (Source/TripSitterUE) should NOT compile its own BeatsyncProcessingTask.cpp
//
// If you need BeatsyncProcessingTask functionality in this module, link against the TripSitterUE
// plugin module or move shared code to a common header-only implementation.
