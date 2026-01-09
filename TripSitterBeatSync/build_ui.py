import unreal

print("=== Building TripSitter BeatSync UI ===")

# Get the widget blueprint
widget_path = "/Game/UI/Widgets/WBP_BeatSyncMain"
widget_bp = unreal.load_asset(widget_path)

if not widget_bp:
    print("ERROR: Widget not found, creating new one...")
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.WidgetBlueprintFactory()
    widget_bp = asset_tools.create_asset("WBP_BeatSyncMain", "/Game/UI/Widgets", unreal.WidgetBlueprint, factory)

# Load textures
wallpaper = unreal.load_asset("/Game/UI/Textures/wallpaper")
header = unreal.load_asset("/Game/UI/Textures/header")

print(f"Wallpaper: {wallpaper}")
print(f"Header: {header}")

# Get the widget tree root
if widget_bp:
    # Access the WidgetTree
    widget_tree = unreal.WidgetBlueprintLibrary.get_all_widgets_of_class(unreal.get_editor_world(), unreal.UserWidget)
    print(f"Widget BP loaded: {widget_bp.get_name()}")

# Create a simple UMG setup using EditorUtilityWidget approach
# Since direct widget tree manipulation is complex, let's set up project defaults

# Configure the Player Controller to show UI on BeginPlay
pc_path = "/Game/Blueprints/BP_BeatSyncPlayerController"
pc_bp = unreal.load_asset(pc_path)

if pc_bp:
    print(f"PlayerController loaded: {pc_bp.get_name()}")
    # Get the blueprint's default object
    pc_class = pc_bp.generated_class()
    if pc_class:
        cdo = unreal.get_default_object(pc_class)
        if cdo:
            # Enable mouse cursor for UI
            cdo.set_editor_property("show_mouse_cursor", True)
            cdo.set_editor_property("enable_click_events", True)
            cdo.set_editor_property("enable_mouse_over_events", True)
            print("Configured PlayerController for UI interaction")

# Set up Game Mode defaults
gm_path = "/Game/Blueprints/BP_BeatSyncGameMode"
gm_bp = unreal.load_asset(gm_path)

if gm_bp and pc_bp:
    gm_class = gm_bp.generated_class()
    pc_class = pc_bp.generated_class()
    if gm_class and pc_class:
        gm_cdo = unreal.get_default_object(gm_class)
        if gm_cdo:
            gm_cdo.set_editor_property("player_controller_class", pc_class)
            gm_cdo.set_editor_property("default_pawn_class", None)  # No pawn needed for UI app
            print("Configured GameMode")

# Save everything
unreal.EditorAssetLibrary.save_asset(widget_path)
unreal.EditorAssetLibrary.save_asset(pc_path)
unreal.EditorAssetLibrary.save_asset(gm_path)

print("")
print("=== Configuration Complete ===")
print("")
print("The widget exists at: Content/UI/Widgets/WBP_BeatSyncMain")
print("Double-click it, then in Designer view:")
print("  1. From Palette, drag 'Canvas Panel' to canvas")
print("  2. Drag 'Image' into the Canvas Panel")
print("  3. Set Image brush to 'wallpaper' texture")
print("")
print("Or create a NEW widget: Right-click in Content Browser")
print("  -> User Interface -> Widget Blueprint")
print("  -> Name it and open Designer tab")
