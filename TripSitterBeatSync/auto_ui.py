import unreal

# This creates a simple full-screen image material that shows wallpaper
# No widget design needed - just a material on a full-screen quad

print("=== Auto UI Setup ===")

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

# Create a Material that displays the wallpaper texture
mat_factory = unreal.MaterialFactoryNew()
mat = asset_tools.create_asset("M_Wallpaper", "/Game/UI", unreal.Material, mat_factory)

if mat:
    print("Created M_Wallpaper material")
    # The material will need to be set up in editor to use the wallpaper texture

# Create a simple HUD blueprint that draws the background
hud_factory = unreal.BlueprintFactory()
hud_factory.set_editor_property("parent_class", unreal.HUD)
hud_bp = asset_tools.create_asset("BP_BeatSyncHUD", "/Game/Blueprints", unreal.Blueprint, hud_factory)

if hud_bp:
    print("Created BP_BeatSyncHUD")

# Update game mode to use our HUD
gm_bp = unreal.load_asset("/Game/Blueprints/BP_BeatSyncGameMode")
if gm_bp and hud_bp:
    gm_class = gm_bp.generated_class()
    hud_class = hud_bp.generated_class()
    if gm_class and hud_class:
        gm_cdo = unreal.get_default_object(gm_class)
        gm_cdo.set_editor_property("hud_class", hud_class)
        print("Set HUD class on GameMode")

unreal.EditorAssetLibrary.save_directory("/Game/Blueprints")
unreal.EditorAssetLibrary.save_directory("/Game/UI")

print("")
print("=== DONE ===")
print("Now in editor, you ONLY need to:")
print("1. Open WBP_BeatSyncMain")
print("2. In the blank canvas, right-click -> search 'image' -> click Image")
print("3. An image appears - in Details on right, find Brush > Image > pick 'wallpaper'")
print("4. Click the Anchor button (looks like a flower) -> pick bottom-right (stretch)")
print("5. Compile and Save")
