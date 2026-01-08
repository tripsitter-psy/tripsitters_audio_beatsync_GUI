#include "VideoPreview.h"
#include <wx/dcbuffer.h>

// FFmpeg
extern "C" {
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "../video/VideoProcessor.h"

wxBEGIN_EVENT_TABLE(VideoPreview, wxPanel)
    EVT_PAINT(VideoPreview::OnPaint)
wxEND_EVENT_TABLE()

VideoPreview::VideoPreview(wxWindow* parent, wxWindowID id,
    const wxPoint& pos, const wxSize& size)
    : wxPanel(parent, id, pos, size)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    SetBackgroundColour(*wxBLACK);
}

void VideoPreview::LoadFrame(const wxString& videoPath, double timestamp) {
    // Attempt to open the video and extract a frame at timestamp using VideoProcessor
    BeatSync::VideoProcessor vp;
    if (!vp.open(videoPath.ToStdString())) {
        // Failed to open - clear preview
        m_frameBitmap = wxBitmap();
        Refresh();
        return;
    }

    if (!vp.seekToTimestamp(timestamp)) {
        m_frameBitmap = wxBitmap();
        Refresh();
        return;
    }

    // Read next frame after seek
    AVFrame* frame = nullptr;
    if (!vp.readFrame(&frame) || !frame) {
        m_frameBitmap = wxBitmap();
        Refresh();
        return;
    }

    int srcW = frame->width;
    int srcH = frame->height;

    // Convert to RGB24
    SwsContext* swsCtx = sws_getContext(
        srcW, srcH, static_cast<AVPixelFormat>(frame->format),
        srcW, srcH, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!swsCtx) {
        m_frameBitmap = wxBitmap();
        Refresh();
        return;
    }

    int rgbLinesize = 3 * srcW;
    int rgbBufSize = av_image_get_buffer_size(AV_PIX_FMT_RGB24, srcW, srcH, 1);
    unsigned char* rgbBuf = static_cast<unsigned char*>(av_malloc(rgbBufSize));
    unsigned char* dstData[4] = { rgbBuf, nullptr, nullptr, nullptr };
    int dstLinesize[4] = { rgbLinesize, 0, 0, 0 };

    sws_scale(swsCtx, frame->data, frame->linesize, 0, srcH, dstData, dstLinesize);

    // Create wxImage (wxImage will take ownership of the buffer if we pass false for static_data)
    wxImage img(srcW, srcH, rgbBuf, false);

    // Convert to bitmap and store
    m_frameBitmap = wxBitmap(img);

    // Cleanup
    sws_freeContext(swsCtx);
    // Note: wxImage took ownership of rgbBuf and will free it; do not free here.

    Refresh();
}

void VideoPreview::Clear() {
    m_frameBitmap = wxBitmap();
    Refresh();
}

void VideoPreview::OnPaint(wxPaintEvent& event) {
    wxAutoBufferedPaintDC dc(this);
    wxSize size = GetSize();
    
    dc.SetBackground(*wxBLACK_BRUSH);
    dc.Clear();
    
    if (m_frameBitmap.IsOk()) {
        // Draw scaled frame
        wxImage img = m_frameBitmap.ConvertToImage();
        img = img.Scale(size.x, size.y, wxIMAGE_QUALITY_HIGH);
        dc.DrawBitmap(wxBitmap(img), 0, 0);
    } else {
        // Draw placeholder
        dc.SetPen(wxPen(wxColour(80, 80, 80), 2));
        dc.SetBrush(*wxTRANSPARENT_BRUSH);
        dc.DrawRectangle(10, 10, size.x - 20, size.y - 20);
        
        dc.SetTextForeground(wxColour(120, 120, 120));
        dc.DrawLabel("Video preview will appear here", 
            wxRect(0, 0, size.x, size.y), wxALIGN_CENTER);
    }
}
