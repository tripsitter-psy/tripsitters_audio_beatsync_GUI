import unreal

# Open the widget blueprint in the editor
widget_path = "/Game/UI/Widgets/WBP_BeatSyncMain"
unreal.EditorAssetLibrary.sync_browser_to_objects([widget_path])
unreal.AssetEditorSubsystem().open_editor_for_assets([unreal.load_asset(widget_path)])
print("Opened WBP_BeatSyncMain in editor")
