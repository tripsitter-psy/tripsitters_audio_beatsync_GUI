#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "BeatsyncSubsystem.h"
#include "BeatSyncWidget.generated.h"

/**
 * Main BeatSync UI Widget - handles all user interaction
 * Design this widget in UMG Editor and bind to these properties/functions
 */
UCLASS()
class TRIPSITTERUE_API UBeatSyncWidget : public UUserWidget
{
	GENERATED_BODY()

public:
	virtual void NativeConstruct() override;
	virtual void NativeDestruct() override;

	// File Selection
	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void BrowseAudioFile();

	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void BrowseVideoFolder();

	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void BrowseOutputFolder();

	// Actions
	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void StartAnalysis();

	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void StartProcessing();

	UFUNCTION(BlueprintCallable, Category = "BeatSync|UI")
	void CancelOperation();

	// Settings
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Settings")
	float ClipDuration = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Settings", meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float BeatSensitivity = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Settings")
	int32 TargetFPS = 30;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BeatSync|Settings")
	FIntPoint TargetResolution = FIntPoint(1920, 1080);

	// Current State (bind these to UI elements)
	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	FString SelectedAudioFile;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	FString SelectedVideoFolder;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	FString SelectedOutputFolder;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	TArray<FString> VideoFiles;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	FBeatData CurrentBeatData;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	float CurrentProgress = 0.0f;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	FString StatusMessage;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	bool bIsProcessing = false;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	bool bHasValidAudio = false;

	UPROPERTY(BlueprintReadOnly, Category = "BeatSync|State")
	bool bHasValidVideos = false;

	// Events for Blueprint binding
	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnAnalysisComplete OnAnalysisCompleteUI;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnProcessingComplete OnProcessingCompleteUI;

	UPROPERTY(BlueprintAssignable, Category = "BeatSync|Events")
	FOnError OnErrorUI;

protected:
	UFUNCTION()
	void HandleAnalysisProgress(float Progress);

	UFUNCTION()
	void HandleProcessingProgress(float Progress);

	UFUNCTION()
	void HandleAnalysisComplete();

	UFUNCTION()
	void HandleProcessingComplete();

	UFUNCTION()
	void HandleError(const FString& ErrorMessage);

	void UpdateVideoFileList();
	UBeatsyncSubsystem* GetBeatsyncSubsystem() const;
};
