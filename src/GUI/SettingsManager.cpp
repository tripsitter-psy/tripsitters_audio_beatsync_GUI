#include "SettingsManager.h"
#include <wx/stdpaths.h>

SettingsManager::SettingsManager() {
    wxString configPath = wxStandardPaths::Get().GetUserDataDir();
    wxFileName::Mkdir(configPath, wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);
    
    m_config = std::make_unique<wxFileConfig>("MTV Trip Sitter", "MTV Trip Sitter",
        configPath + "/settings.ini");
}

SettingsManager::~SettingsManager() {
    if (m_config) {
        m_config->Flush();
    }
}

wxString SettingsManager::GetString(const wxString& key, const wxString& defaultValue) {
    return m_config->Read(key, defaultValue);
}

int SettingsManager::GetInt(const wxString& key, int defaultValue) {
    return m_config->ReadLong(key, defaultValue);
}

bool SettingsManager::GetBool(const wxString& key, bool defaultValue) {
    return m_config->ReadBool(key, defaultValue);
}

void SettingsManager::SetString(const wxString& key, const wxString& value) {
    m_config->Write(key, value);
}

void SettingsManager::SetInt(const wxString& key, int value) {
    m_config->Write(key, (long)value);
}

void SettingsManager::SetBool(const wxString& key, bool value) {
    m_config->Write(key, value);
}
