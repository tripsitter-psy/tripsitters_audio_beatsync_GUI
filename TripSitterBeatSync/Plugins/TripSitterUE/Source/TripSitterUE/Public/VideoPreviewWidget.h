#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "MediaPlayer.h"
#include "MediaTexture.h"
#include "VideoPreviewWidget.generated.h"

/**
 * Widget for previewing video files with playback controls
 */
UCLASS()
class TRIPSITTERUE_API UVideoPreviewWidget : public UUserWidget
{
	GENERATED_BODY()

public:
	virtual void NativeConstruct() override;
	virtual void NativeDestruct() override;
	virtual void NativeTick(const FGeometry& MyGeometry, float InDeltaTime) override;

	/** Load and preview a video file */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	bool LoadVideo(const FString& FilePath);

	/** Play the loaded video */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	void Play();

	/** Pause playback */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	void Pause();

	/** Stop and reset to beginning */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	void Stop();

	/** Seek to specific time */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	void SeekToTime(float TimeInSeconds);

	/** Get current playback time */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	float GetCurrentTime() const;

	/** Get video duration */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	float GetDuration() const;

	/** Check if video is playing */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	bool IsPlaying() const;

	/** Get the media texture for display */
	UFUNCTION(BlueprintCallable, Category = "BeatSync|VideoPreview")
	UMediaTexture* GetMediaTexture() const { return MediaTexture; }

	// State
	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|VideoPreview|State")
	FString CurrentVideoPath;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|VideoPreview|State")
	bool bIsVideoLoaded = false;

	// Events
	DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPlaybackTimeChanged, float, TimeInSeconds);

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|VideoPreview|Events")
	FOnPlaybackTimeChanged OnPlaybackTimeChanged;

	DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnVideoLoaded);

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|VideoPreview|Events")
	FOnVideoLoaded OnVideoLoaded;

	DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnVideoEnded);

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|VideoPreview|Events")
	FOnVideoEnded OnVideoEnded;

protected:
	UPROPERTY()
	UMediaPlayer* MediaPlayer;

	UPROPERTY()
	UMediaTexture* MediaTexture;

	float LastReportedTime = -1.0f;

	UFUNCTION()
	void HandleMediaOpened(FString OpenedUrl);

	UFUNCTION()
	void HandleMediaOpenFailed(FString FailedUrl);

	UFUNCTION()
	void HandleEndReached();
};
