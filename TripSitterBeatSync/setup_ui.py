import unreal
import os

# Asset paths
project_dir = unreal.Paths.project_dir()
textures_path = os.path.join(project_dir, "Content", "UI", "Textures")

print("=== TripSitter BeatSync UI Setup ===")

# 1. Import textures
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
import_tasks = []

texture_files = ["wallpaper.png", "background.png", "header.png"]
for tex_file in texture_files:
    full_path = os.path.join(textures_path, tex_file)
    if os.path.exists(full_path):
        task = unreal.AssetImportTask()
        task.filename = full_path
        task.destination_path = "/Game/UI/Textures"
        task.automated = True
        task.save = True
        task.replace_existing = True
        import_tasks.append(task)
        print(f"Queued import: {tex_file}")
    else:
        print(f"Warning: {tex_file} not found at {full_path}")

if import_tasks:
    asset_tools.import_asset_tasks(import_tasks)
    print("Textures imported!")

# 2. Create folders
editor_asset_lib = unreal.EditorAssetLibrary
folders = ["/Game/UI", "/Game/UI/Textures", "/Game/UI/Widgets", "/Game/Maps", "/Game/Blueprints"]
for folder in folders:
    if not editor_asset_lib.does_directory_exist(folder):
        editor_asset_lib.make_directory(folder)
        print(f"Created folder: {folder}")

# 3. Create a basic level and save it
editor_level_lib = unreal.EditorLevelLibrary
world = editor_level_lib.get_editor_world()

# Save current level as MainMenu
level_path = "/Game/Maps/MainMenu"
if not editor_asset_lib.does_asset_exist(level_path):
    unreal.EditorLevelLibrary.new_level("/Engine/Maps/Templates/OpenWorld")
    unreal.EditorLevelLibrary.save_current_level()
    print("Created MainMenu level")

# 4. Set project settings for UI-focused app
print("")
print("=== Setup Complete! ===")
print("")
print("Next steps:")
print("1. Right-click in Content/UI/Widgets -> User Interface -> Widget Blueprint")
print("2. Name it 'WBP_MainMenu'")
print("3. In the Widget, add an Image and set it to 'wallpaper' texture")
print("4. Add buttons/text for your BeatSync UI")
print("5. Create a Blueprint -> Game Mode Base called 'BP_GameMode'")
print("6. In Level Blueprint: BeginPlay -> Create Widget -> Add to Viewport")
print("")
print("Your textures are in: /Game/UI/Textures/")
