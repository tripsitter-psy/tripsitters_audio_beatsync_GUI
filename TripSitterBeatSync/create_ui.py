import unreal

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

print("=== Creating TripSitter BeatSync UI ===")

# Create Widget Blueprint
widget_factory = unreal.WidgetBlueprintFactory()
widget_bp = asset_tools.create_asset("WBP_BeatSyncMain", "/Game/UI/Widgets", unreal.WidgetBlueprint, widget_factory)
if widget_bp:
    print("Created: /Game/UI/Widgets/WBP_BeatSyncMain")

# Create Game Mode Blueprint
gm_factory = unreal.BlueprintFactory()
gm_factory.set_editor_property("parent_class", unreal.GameModeBase)
gm_bp = asset_tools.create_asset("BP_BeatSyncGameMode", "/Game/Blueprints", unreal.Blueprint, gm_factory)
if gm_bp:
    print("Created: /Game/Blueprints/BP_BeatSyncGameMode")

# Create Player Controller
pc_factory = unreal.BlueprintFactory()
pc_factory.set_editor_property("parent_class", unreal.PlayerController)
pc_bp = asset_tools.create_asset("BP_BeatSyncPlayerController", "/Game/Blueprints", unreal.Blueprint, pc_factory)
if pc_bp:
    print("Created: /Game/Blueprints/BP_BeatSyncPlayerController")

# Create a new map
unreal.EditorLevelLibrary.new_level("/Engine/Maps/Templates/Template_Default")
unreal.EditorLevelLibrary.save_current_level()
unreal.EditorAssetLibrary.rename_asset("/Game/Untitled", "/Game/Maps/MainMenu")

# Save all
unreal.EditorAssetLibrary.save_directory("/Game/UI")
unreal.EditorAssetLibrary.save_directory("/Game/Blueprints")
unreal.EditorAssetLibrary.save_directory("/Game/Maps")

print("=== Done! Refresh Content Browser in Editor ===")
