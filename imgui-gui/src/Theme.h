// TripSitter neon theme - ports the color palette from STripSitterMainWidget
// (neon cyan / purple / hot-pink / green on a dark blue background) into an
// ImGui style.
#pragma once

#include "imgui.h"

namespace Theme
{
    // Palette (matches the Slate FLinearColor values).
    constexpr ImVec4 NeonCyan   = ImVec4(0.000f, 0.851f, 1.000f, 1.0f);
    constexpr ImVec4 NeonPurple = ImVec4(0.545f, 0.000f, 1.000f, 1.0f);
    constexpr ImVec4 DarkBg     = ImVec4(0.039f, 0.039f, 0.102f, 1.0f);
    constexpr ImVec4 ControlBg  = ImVec4(0.078f, 0.078f, 0.157f, 1.0f);
    constexpr ImVec4 TextColor  = ImVec4(0.784f, 0.863f, 1.000f, 1.0f);
    constexpr ImVec4 HotPink    = ImVec4(1.000f, 0.000f, 0.502f, 1.0f);
    constexpr ImVec4 NeonGreen  = ImVec4(0.000f, 1.000f, 0.392f, 1.0f);

    inline ImVec4 WithAlpha(const ImVec4& c, float a) { return ImVec4(c.x, c.y, c.z, a); }

    inline void Apply()
    {
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding    = 8.0f;
        s.FrameRounding     = 6.0f;
        s.GrabRounding      = 6.0f;
        s.PopupRounding     = 6.0f;
        s.ScrollbarRounding = 8.0f;
        s.FramePadding      = ImVec2(8, 5);
        s.ItemSpacing       = ImVec2(10, 8);
        s.WindowPadding     = ImVec2(14, 12);
        s.WindowBorderSize  = 0.0f;
        s.FrameBorderSize   = 1.0f;

        ImVec4* c = s.Colors;
        c[ImGuiCol_Text]            = TextColor;
        c[ImGuiCol_TextDisabled]    = WithAlpha(TextColor, 0.45f);
        c[ImGuiCol_WindowBg]        = DarkBg;
        c[ImGuiCol_ChildBg]         = WithAlpha(ControlBg, 0.55f);
        c[ImGuiCol_PopupBg]         = ControlBg;
        c[ImGuiCol_Border]          = WithAlpha(NeonCyan, 0.30f);
        c[ImGuiCol_FrameBg]         = ControlBg;
        c[ImGuiCol_FrameBgHovered]  = WithAlpha(NeonCyan, 0.25f);
        c[ImGuiCol_FrameBgActive]   = WithAlpha(NeonPurple, 0.40f);
        c[ImGuiCol_TitleBg]         = ControlBg;
        c[ImGuiCol_TitleBgActive]   = WithAlpha(NeonPurple, 0.55f);
        c[ImGuiCol_Header]          = WithAlpha(NeonPurple, 0.45f);
        c[ImGuiCol_HeaderHovered]   = WithAlpha(NeonCyan, 0.45f);
        c[ImGuiCol_HeaderActive]    = WithAlpha(NeonCyan, 0.65f);
        c[ImGuiCol_Button]          = WithAlpha(NeonPurple, 0.55f);
        c[ImGuiCol_ButtonHovered]   = WithAlpha(NeonCyan, 0.55f);
        c[ImGuiCol_ButtonActive]    = WithAlpha(HotPink, 0.70f);
        c[ImGuiCol_CheckMark]       = NeonGreen;
        c[ImGuiCol_SliderGrab]      = NeonCyan;
        c[ImGuiCol_SliderGrabActive]= HotPink;
        c[ImGuiCol_SeparatorHovered]= NeonCyan;
        c[ImGuiCol_Tab]             = WithAlpha(NeonPurple, 0.40f);
        c[ImGuiCol_TabHovered]      = WithAlpha(NeonCyan, 0.55f);
        c[ImGuiCol_TabActive]       = WithAlpha(NeonCyan, 0.40f);
        c[ImGuiCol_PlotHistogram]   = NeonCyan;
        c[ImGuiCol_PlotHistogramHovered] = HotPink;
    }
}
