# Transition Shaders

This directory holds GLSL transition shaders compatible with [gl-transitions](https://gl-transitions.com/).

## How to add new transitions
1. Copy a `.glsl` file into this folder.
2. Add metadata comments at the top (see `fade.glsl` as a template).
3. TransitionLibrary will auto-discover shaders at runtime.

## Categories
- **blend**: fade, dissolve, crossfade
- **geometric**: wipe, slide, circle, radial
- **distortion**: pixelize, blur, squeeze
- **psychedelic**: kaleidoscope, fractal, morph
