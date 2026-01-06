// Placeholder fade transition shader
// Author: gl-transitions community
// Category: blend
// Name: fade

vec4 transition(vec2 uv) {
    return mix(getFromColor(uv), getToColor(uv), progress);
}
