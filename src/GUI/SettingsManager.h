#pragma once
#include <wx/wx.h>
#include <wx/fileconf.h>
#include <memory>

class SettingsManager {
public:
    SettingsManager();
    ~SettingsManager();
    
    wxString GetString(const wxString& key, const wxString& defaultValue = "");
    int GetInt(const wxString& key, int defaultValue = 0);
    bool GetBool(const wxString& key, bool defaultValue = false);
    
    void SetString(const wxString& key, const wxString& value);
    void SetInt(const wxString& key, int value);
    void SetBool(const wxString& key, bool value);
    
private:
    std::unique_ptr<wxFileConfig> m_config;
};
