#include "BeatsyncLoader.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"

// C API function pointer types
extern "C"
{
	typedef const char* (*bs_resolve_ffmpeg_path_t)();
	typedef void* (*bs_create_audio_analyzer_t)();
	typedef void (*bs_destroy_audio_analyzer_t)(void*);
	typedef void* (*bs_create_video_writer_t)();
	typedef void (*bs_destroy_video_writer_t)(void*);
	typedef const char* (*bs_video_get_last_error_t)(void*);
	typedef void (*bs_progress_cb)(double progress, void* user_data);
	typedef void (*bs_video_set_progress_callback_t)(void*, bs_progress_cb, void*);

	typedef struct {
		double* beats;
		size_t count;
		double bpm;
		double duration;
	} bs_beatgrid_t;

	typedef int (*bs_analyze_audio_t)(void*, const char*, bs_beatgrid_t*);
	typedef void (*bs_free_beatgrid_t)(bs_beatgrid_t*);
	typedef int (*bs_video_cut_at_beats_t)(void*, const char*, const double*, size_t, const char*, double);
	typedef int (*bs_video_concatenate_t)(const char**, size_t, const char*);
}

struct FBeatsyncApi
{
	void* DllHandle = nullptr;
	bs_resolve_ffmpeg_path_t resolve_ffmpeg = nullptr;
	bs_create_audio_analyzer_t create_analyzer = nullptr;
	bs_destroy_audio_analyzer_t destroy_analyzer = nullptr;
	bs_analyze_audio_t analyze_audio = nullptr;
	bs_free_beatgrid_t free_beatgrid = nullptr;
	bs_create_video_writer_t create_video_writer = nullptr;
	bs_destroy_video_writer_t destroy_video_writer = nullptr;
	bs_video_get_last_error_t video_get_last_error = nullptr;
	bs_video_set_progress_callback_t video_set_progress_callback = nullptr;
	bs_video_cut_at_beats_t video_cut_at_beats = nullptr;
	bs_video_concatenate_t video_concatenate = nullptr;
};

static FBeatsyncApi GApi;

bool FBeatsyncLoader::Initialize()
{
	if (GApi.DllHandle) return true;

	FString Filename;
	FString Subdir;
#if PLATFORM_WINDOWS
	Filename = TEXT("beatsync_backend.dll");
	Subdir = TEXT("x64");
#elif PLATFORM_MAC
	Filename = TEXT("libbeatsync_backend.dylib");
	Subdir = TEXT("Mac");
#else
	Filename = TEXT("libbeatsync_backend.so");
	Subdir = TEXT("Linux");
#endif

	// Try multiple paths to find the library
	TArray<FString> SearchPaths;

	// 1. ThirdParty relative to project dir
	SearchPaths.Add(FPaths::Combine(FPaths::ProjectDir(), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename));

	// 2. Try plugin directory
	if (IPluginManager::Get().FindPlugin(TEXT("TripSitterUE")))
	{
		FString PluginDir = IPluginManager::Get().FindPlugin(TEXT("TripSitterUE"))->GetBaseDir();
		SearchPaths.Add(FPaths::Combine(PluginDir, TEXT(".."), TEXT(".."), TEXT("ThirdParty"), TEXT("beatsync"), TEXT("lib"), Subdir, Filename));
	}

	// 3. For packaged builds: relative to executable/base dir
	FString BaseDir = FPlatformProcess::BaseDir();
	SearchPaths.Add(FPaths::Combine(BaseDir, TEXT("../UE/TripSitterBeatSync/ThirdParty/beatsync/lib"), Subdir, Filename));
	SearchPaths.Add(FPaths::Combine(BaseDir, TEXT("../../UE/TripSitterBeatSync/ThirdParty/beatsync/lib"), Subdir, Filename));

	// 4. Check inside app bundle on Mac
#if PLATFORM_MAC
	SearchPaths.Add(FPaths::Combine(BaseDir, TEXT("../Resources/ThirdParty/beatsync/lib"), Subdir, Filename));
	// Contents/UE/TripSitterBeatSync/ThirdParty path from MacOS folder
	SearchPaths.Add(FPaths::Combine(BaseDir, TEXT("../UE/TripSitterBeatSync/ThirdParty/beatsync/lib/Mac"), Filename));
#endif

	FString DllPath;
	for (const FString& Path : SearchPaths)
	{
		FString NormalizedPath = FPaths::ConvertRelativePathToFull(Path);
		UE_LOG(LogTemp, Log, TEXT("Checking for Beatsync library at: %s"), *NormalizedPath);
		if (FPaths::FileExists(NormalizedPath))
		{
			DllPath = NormalizedPath;
			break;
		}
	}

	if (DllPath.IsEmpty())
	{
		UE_LOG(LogTemp, Error, TEXT("Beatsync library not found in any search path"));
		return false;
	}

	GApi.DllHandle = FPlatformProcess::GetDllHandle(*DllPath);
	if (!GApi.DllHandle)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to load Beatsync library: %s"), *DllPath);
		return false;
	}

	// Load all function pointers
	GApi.resolve_ffmpeg = (bs_resolve_ffmpeg_path_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_resolve_ffmpeg_path"));
	GApi.create_analyzer = (bs_create_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_audio_analyzer"));
	GApi.destroy_analyzer = (bs_destroy_audio_analyzer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_audio_analyzer"));
	GApi.analyze_audio = (bs_analyze_audio_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_analyze_audio"));
	GApi.free_beatgrid = (bs_free_beatgrid_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_free_beatgrid"));
	GApi.create_video_writer = (bs_create_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_create_video_writer"));
	GApi.destroy_video_writer = (bs_destroy_video_writer_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_destroy_video_writer"));
	GApi.video_get_last_error = (bs_video_get_last_error_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_get_last_error"));
	GApi.video_set_progress_callback = (bs_video_set_progress_callback_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_set_progress_callback"));
	GApi.video_cut_at_beats = (bs_video_cut_at_beats_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_cut_at_beats"));
	GApi.video_concatenate = (bs_video_concatenate_t)FPlatformProcess::GetDllExport(GApi.DllHandle, TEXT("bs_video_concatenate"));

	// Check required symbols
	if (!GApi.resolve_ffmpeg || !GApi.create_analyzer || !GApi.destroy_analyzer || !GApi.analyze_audio || !GApi.free_beatgrid)
	{
		UE_LOG(LogTemp, Error, TEXT("Required symbols not found in Beatsync DLL"));
		FPlatformProcess::FreeDllHandle(GApi.DllHandle);
		GApi.DllHandle = nullptr;
		return false;
	}

	UE_LOG(LogTemp, Log, TEXT("Beatsync library loaded: %s"), *DllPath);
	return true;
}

void FBeatsyncLoader::Shutdown()
{
	if (GApi.DllHandle)
	{
		FPlatformProcess::FreeDllHandle(GApi.DllHandle);
		FMemory::Memzero(&GApi, sizeof(GApi));
	}
}

bool FBeatsyncLoader::IsInitialized()
{
	return GApi.DllHandle != nullptr;
}

FString FBeatsyncLoader::ResolveFFmpegPath()
{
	if (!GApi.resolve_ffmpeg) return FString();
	const char* Path = GApi.resolve_ffmpeg();
	return Path ? FString(UTF8_TO_TCHAR(Path)) : FString();
}

void* FBeatsyncLoader::CreateAnalyzer()
{
	if (!GApi.create_analyzer) return nullptr;
	return GApi.create_analyzer();
}

void FBeatsyncLoader::DestroyAnalyzer(void* Handle)
{
	if (Handle && GApi.destroy_analyzer)
	{
		GApi.destroy_analyzer(Handle);
	}
}

void* FBeatsyncLoader::CreateVideoWriter()
{
	if (!GApi.create_video_writer) return nullptr;
	return GApi.create_video_writer();
}

void FBeatsyncLoader::DestroyVideoWriter(void* Handle)
{
	if (Handle && GApi.destroy_video_writer)
	{
		GApi.destroy_video_writer(Handle);
	}
}

FString FBeatsyncLoader::GetVideoWriterLastError(void* Handle)
{
	if (!Handle || !GApi.video_get_last_error) return FString();
	const char* Err = GApi.video_get_last_error(Handle);
	return Err ? FString(UTF8_TO_TCHAR(Err)) : FString();
}

bool FBeatsyncLoader::AnalyzeAudio(void* AnalyzerHandle, const FString& FilePath, TArray<double>& OutBeats, double& OutBPM, double& OutDuration)
{
	if (!AnalyzerHandle || !GApi.analyze_audio || !GApi.free_beatgrid) return false;

	bs_beatgrid_t Grid = {};
	FString NormalizedPath = FPaths::ConvertRelativePathToFull(FilePath);

	int Result = GApi.analyze_audio(AnalyzerHandle, TCHAR_TO_UTF8(*NormalizedPath), &Grid);
	if (Result != 0)
	{
		return false;
	}

	OutBeats.Empty(Grid.count);
	for (size_t i = 0; i < Grid.count; ++i)
	{
		OutBeats.Add(Grid.beats[i]);
	}
	OutBPM = Grid.bpm;
	OutDuration = Grid.duration;

	GApi.free_beatgrid(&Grid);
	return true;
}

bool FBeatsyncLoader::CutVideoAtBeats(void* WriterHandle, const FString& InputVideo, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration)
{
	if (!WriterHandle || !GApi.video_cut_at_beats) return false;

	FString NormalizedInput = FPaths::ConvertRelativePathToFull(InputVideo);
	FString NormalizedOutput = FPaths::ConvertRelativePathToFull(OutputVideo);

	int Result = GApi.video_cut_at_beats(
		WriterHandle,
		TCHAR_TO_UTF8(*NormalizedInput),
		BeatTimes.GetData(),
		BeatTimes.Num(),
		TCHAR_TO_UTF8(*NormalizedOutput),
		ClipDuration
	);

	return Result == 0;
}

bool FBeatsyncLoader::ConcatenateVideos(const TArray<FString>& InputVideos, const FString& OutputVideo)
{
	if (!GApi.video_concatenate) return false;

	TArray<FString> NormalizedPaths;
	TArray<const char*> PathPtrs;
	TArray<TArray<char>> PathBuffers;

	for (const FString& Path : InputVideos)
	{
		FString Normalized = FPaths::ConvertRelativePathToFull(Path);
		TArray<char>& Buffer = PathBuffers.AddDefaulted_GetRef();
		FTCHARToUTF8 Converter(*Normalized);
		Buffer.SetNumUninitialized(Converter.Length() + 1);
		FMemory::Memcpy(Buffer.GetData(), Converter.Get(), Converter.Length() + 1);
		PathPtrs.Add(Buffer.GetData());
	}

	FString NormalizedOutput = FPaths::ConvertRelativePathToFull(OutputVideo);

	int Result = GApi.video_concatenate(
		PathPtrs.GetData(),
		PathPtrs.Num(),
		TCHAR_TO_UTF8(*NormalizedOutput)
	);

	return Result == 0;
}
