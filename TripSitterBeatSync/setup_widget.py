import unreal

print("=== Setting up BeatSync Widget ===")

# Load the widget blueprint
widget_path = "/Game/UI/Widgets/WBP_BeatSyncMain"
widget_bp = unreal.EditorAssetLibrary.load_asset(widget_path)

if widget_bp:
    print(f"Loaded widget: {widget_path}")

    # Load wallpaper texture
    wallpaper = unreal.EditorAssetLibrary.load_asset("/Game/UI/Textures/wallpaper")
    if wallpaper:
        print("Wallpaper texture loaded")

    # Load the game mode and set default classes
    gm_path = "/Game/Blueprints/BP_BeatSyncGameMode"
    gm_bp = unreal.EditorAssetLibrary.load_asset(gm_path)

    pc_path = "/Game/Blueprints/BP_BeatSyncPlayerController"
    pc_bp = unreal.EditorAssetLibrary.load_asset(pc_path)

    if gm_bp:
        # Set player controller class on game mode
        gm_cdo = unreal.get_default_object(gm_bp.generated_class())
        if gm_cdo and pc_bp:
            gm_cdo.set_editor_property("player_controller_class", pc_bp.generated_class())
            print("Set PlayerController on GameMode")

# Update project settings to use our game mode
project_settings = unreal.GameMapsSettings.get_default_object()
project_settings.set_editor_property("global_default_game_mode", "/Game/Blueprints/BP_BeatSyncGameMode.BP_BeatSyncGameMode_C")

# Create and save a map
map_package = "/Game/Maps/MainMenu"
if not unreal.EditorAssetLibrary.does_asset_exist(map_package):
    # Create empty level
    world = unreal.EditorLevelLibrary.get_editor_world()
    success = unreal.EditorLoadingAndSavingUtils.save_map(world, "/Game/Maps/MainMenu")
    print(f"Saved map: {success}")

unreal.EditorAssetLibrary.save_directory("/Game/Blueprints")
unreal.EditorAssetLibrary.save_directory("/Game/Maps")

print("")
print("=== NEXT: In UE Editor ===")
print("1. Open WBP_BeatSyncMain (Content/UI/Widgets)")
print("2. In Designer, add Canvas Panel as root")
print("3. Add Image widget, anchor to fill screen")
print("4. In Image Details, set Brush > Image to 'wallpaper'")
print("5. Open BP_BeatSyncPlayerController")
print("6. In Event Graph: BeginPlay -> Create Widget (WBP_BeatSyncMain) -> Add to Viewport")
print("7. Play to test!")
