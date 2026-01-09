#pragma once

#include "CoreMinimal.h"

/**
 * Loads and manages the beatsync backend library (DLL/dylib/so)
 * Provides access to audio analysis and video processing functions
 */
class TRIPSITTERUE_API FBeatsyncLoader
{
public:
	/** Initialize the beatsync library - call once at module startup */
	static bool Initialize();

	/** Shutdown and unload the library */
	static void Shutdown();

	/** Check if the library is loaded and ready */
	static bool IsInitialized();

	/** Get the path to FFmpeg executable */
	static FString ResolveFFmpegPath();

	/** Create an audio analyzer instance - caller must call DestroyAnalyzer when done */
	static void* CreateAnalyzer();

	/** Destroy an audio analyzer instance */
	static void DestroyAnalyzer(void* Handle);

	/** Create a video writer instance - caller must call DestroyVideoWriter when done */
	static void* CreateVideoWriter();

	/** Destroy a video writer instance */
	static void DestroyVideoWriter(void* Handle);

	/** Get last error from video writer */
	static FString GetVideoWriterLastError(void* Handle);

	/**
	 * Analyze audio file and detect beats
	 * @param AnalyzerHandle Handle from CreateAnalyzer()
	 * @param FilePath Path to audio file
	 * @param OutBeats Array to receive beat timestamps (in seconds)
	 * @param OutBPM Detected BPM
	 * @param OutDuration Audio duration in seconds
	 * @return true on success
	 */
	static bool AnalyzeAudio(void* AnalyzerHandle, const FString& FilePath, TArray<double>& OutBeats, double& OutBPM, double& OutDuration);

	/**
	 * Cut video at beat timestamps
	 * @param WriterHandle Handle from CreateVideoWriter()
	 * @param InputVideo Path to input video
	 * @param BeatTimes Array of beat timestamps
	 * @param OutputVideo Path for output video
	 * @param ClipDuration Duration of each clip in seconds
	 * @return true on success
	 */
	static bool CutVideoAtBeats(void* WriterHandle, const FString& InputVideo, const TArray<double>& BeatTimes, const FString& OutputVideo, double ClipDuration);

	/**
	 * Concatenate multiple videos
	 * @param InputVideos Array of input video paths
	 * @param OutputVideo Path for output video
	 * @return true on success
	 */
	static bool ConcatenateVideos(const TArray<FString>& InputVideos, const FString& OutputVideo);
};
