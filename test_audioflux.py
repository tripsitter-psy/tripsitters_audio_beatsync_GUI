"""Quick test script to debug AudioFlux beat detection"""
import ctypes
import os
import sys

# Load DLL
dll_path = r"C:\UE5_Source\UnrealEngine\Engine\Binaries\Win64\beatsync_backend_shared.dll"
print(f"Loading DLL from: {dll_path}")

if not os.path.exists(dll_path):
    print(f"ERROR: DLL not found at {dll_path}")
    sys.exit(1)

dll = ctypes.CDLL(dll_path)

# Check AudioFlux availability
dll.bs_audioflux_is_available.restype = ctypes.c_int
available = dll.bs_audioflux_is_available()
print(f"AudioFlux available: {available}")

if not available:
    print("ERROR: AudioFlux is not available in the DLL!")
    sys.exit(1)

# Define the result structure
class bs_segment_t(ctypes.Structure):
    _fields_ = [
        ("start_time", ctypes.c_double),
        ("end_time", ctypes.c_double),
        ("label", ctypes.c_char_p),
    ]

class bs_ai_result_t(ctypes.Structure):
    _fields_ = [
        ("beats", ctypes.POINTER(ctypes.c_double)),
        ("beat_count", ctypes.c_size_t),
        ("downbeats", ctypes.POINTER(ctypes.c_double)),
        ("downbeat_count", ctypes.c_size_t),
        ("bpm", ctypes.c_double),
        ("duration", ctypes.c_double),
        ("segments", ctypes.POINTER(bs_segment_t)),
        ("segment_count", ctypes.c_size_t),
    ]

# Progress callback type
PROGRESS_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_float, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))

def progress_callback(progress, stage, message, user_data):
    stage_str = stage.decode('utf-8') if stage else ""
    msg_str = message.decode('utf-8') if message else ""
    print(f"  Progress: {progress*100:.1f}% - {stage_str} {msg_str}")
    return 1  # Continue

progress_func = PROGRESS_CB(progress_callback)

# Set up function signature
dll.bs_audioflux_analyze.argtypes = [ctypes.c_char_p, ctypes.POINTER(bs_ai_result_t), PROGRESS_CB, ctypes.c_void_p]
dll.bs_audioflux_analyze.restype = ctypes.c_int

dll.bs_free_ai_result.argtypes = [ctypes.POINTER(bs_ai_result_t)]
dll.bs_free_ai_result.restype = None

dll.bs_ai_get_last_error.restype = ctypes.c_char_p

# Get a test audio file path - use a file the user has tested with
print("\nPlease drag and drop an audio file onto this window, or type the path:")
audio_path = input("Audio file path: ").strip().strip('"')

if not os.path.exists(audio_path):
    print(f"ERROR: Audio file not found: {audio_path}")
    sys.exit(1)

print(f"\nAnalyzing: {audio_path}")
print("-" * 60)

result = bs_ai_result_t()
ret = dll.bs_audioflux_analyze(audio_path.encode('utf-8'), ctypes.byref(result), progress_func, None)

print("-" * 60)
print(f"Return code: {ret}")

if ret != 0:
    error = dll.bs_ai_get_last_error()
    if error:
        print(f"Error: {error.decode('utf-8')}")
else:
    print(f"Beats found: {result.beat_count}")
    print(f"BPM: {result.bpm:.2f}")
    print(f"Duration: {result.duration:.2f}s")

    if result.beat_count > 0 and result.beat_count < 20:
        print("First beats (seconds):")
        for i in range(min(10, result.beat_count)):
            print(f"  {i+1}: {result.beats[i]:.3f}s")

    dll.bs_free_ai_result(ctypes.byref(result))

# Show debug log
print("\n" + "=" * 60)
print("DEBUG LOG:")
print("=" * 60)
log_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'beatsync_debug.log')
if os.path.exists(log_path):
    with open(log_path, 'r') as f:
        # Show last 50 lines
        lines = f.readlines()
        for line in lines[-50:]:
            print(line.rstrip())
else:
    print(f"Debug log not found at: {log_path}")

input("\nPress Enter to exit...")
