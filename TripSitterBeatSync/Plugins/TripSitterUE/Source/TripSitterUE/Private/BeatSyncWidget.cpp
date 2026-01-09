#include "BeatSyncWidget.h"
#include "Framework/Application/SlateApplication.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Kismet/GameplayStatics.h"

#if WITH_DESKTOP_PLATFORM
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#endif

void UBeatSyncWidget::NativeConstruct()
{
	Super::NativeConstruct();

	// Bind to subsystem events
	if (UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem())
	{
		Subsystem->OnAnalysisProgress.AddDynamic(this, &UBeatSyncWidget::HandleAnalysisProgress);
		Subsystem->OnProcessingProgress.AddDynamic(this, &UBeatSyncWidget::HandleProcessingProgress);
		Subsystem->OnAnalysisComplete.AddDynamic(this, &UBeatSyncWidget::HandleAnalysisComplete);
		Subsystem->OnProcessingComplete.AddDynamic(this, &UBeatSyncWidget::HandleProcessingComplete);
		Subsystem->OnError.AddDynamic(this, &UBeatSyncWidget::HandleError);
	}

	StatusMessage = TEXT("Ready - Select an audio file to begin");
}

void UBeatSyncWidget::NativeDestruct()
{
	// Unbind from subsystem events
	if (UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem())
	{
		Subsystem->OnAnalysisProgress.RemoveDynamic(this, &UBeatSyncWidget::HandleAnalysisProgress);
		Subsystem->OnProcessingProgress.RemoveDynamic(this, &UBeatSyncWidget::HandleProcessingProgress);
		Subsystem->OnAnalysisComplete.RemoveDynamic(this, &UBeatSyncWidget::HandleAnalysisComplete);
		Subsystem->OnProcessingComplete.RemoveDynamic(this, &UBeatSyncWidget::HandleProcessingComplete);
		Subsystem->OnError.RemoveDynamic(this, &UBeatSyncWidget::HandleError);
	}

	Super::NativeDestruct();
}

UBeatsyncSubsystem* UBeatSyncWidget::GetBeatsyncSubsystem() const
{
	if (UGameInstance* GI = UGameplayStatics::GetGameInstance(this))
	{
		return GI->GetSubsystem<UBeatsyncSubsystem>();
	}
	return nullptr;
}

void UBeatSyncWidget::BrowseAudioFile()
{
#if WITH_DESKTOP_PLATFORM
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		return;
	}

	TArray<FString> OutFiles;
	const FString FileTypes = TEXT("Audio Files (*.mp3;*.wav;*.flac;*.ogg)|*.mp3;*.wav;*.flac;*.ogg|All Files (*.*)|*.*");

	void* ParentWindowHandle = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
		? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
		: nullptr;

	bool bOpened = DesktopPlatform->OpenFileDialog(
		ParentWindowHandle,
		TEXT("Select Audio File"),
		FPaths::GetProjectFilePath(),
		TEXT(""),
		FileTypes,
		EFileDialogFlags::None,
		OutFiles
	);

	if (bOpened && OutFiles.Num() > 0)
	{
		SelectedAudioFile = OutFiles[0];
		bHasValidAudio = true;
		StatusMessage = FString::Printf(TEXT("Audio: %s"), *FPaths::GetCleanFilename(SelectedAudioFile));
	}
#else
	StatusMessage = TEXT("File browsing not available in this build");
#endif
}

void UBeatSyncWidget::BrowseVideoFolder()
{
#if WITH_DESKTOP_PLATFORM
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		return;
	}

	FString OutFolder;
	void* ParentWindowHandle = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
		? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
		: nullptr;

	bool bOpened = DesktopPlatform->OpenDirectoryDialog(
		ParentWindowHandle,
		TEXT("Select Video Folder"),
		FPaths::GetProjectFilePath(),
		OutFolder
	);

	if (bOpened)
	{
		SelectedVideoFolder = OutFolder;
		UpdateVideoFileList();
	}
#else
	StatusMessage = TEXT("File browsing not available in this build");
#endif
}

void UBeatSyncWidget::BrowseOutputFolder()
{
#if WITH_DESKTOP_PLATFORM
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		return;
	}

	FString OutFolder;
	void* ParentWindowHandle = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
		? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
		: nullptr;

	bool bOpened = DesktopPlatform->OpenDirectoryDialog(
		ParentWindowHandle,
		TEXT("Select Output Folder"),
		FPaths::GetProjectFilePath(),
		OutFolder
	);

	if (bOpened)
	{
		SelectedOutputFolder = OutFolder;
		StatusMessage = FString::Printf(TEXT("Output: %s"), *SelectedOutputFolder);
	}
#else
	StatusMessage = TEXT("File browsing not available in this build");
#endif
}

void UBeatSyncWidget::UpdateVideoFileList()
{
	VideoFiles.Empty();

	if (SelectedVideoFolder.IsEmpty())
	{
		bHasValidVideos = false;
		return;
	}

	// Find all video files in the folder
	TArray<FString> Extensions = { TEXT("*.mp4"), TEXT("*.avi"), TEXT("*.mov"), TEXT("*.mkv"), TEXT("*.webm") };

	for (const FString& Ext : Extensions)
	{
		TArray<FString> FoundFiles;
		IFileManager::Get().FindFiles(FoundFiles, *FPaths::Combine(SelectedVideoFolder, Ext), true, false);

		for (const FString& File : FoundFiles)
		{
			VideoFiles.Add(FPaths::Combine(SelectedVideoFolder, File));
		}
	}

	bHasValidVideos = VideoFiles.Num() > 0;
	StatusMessage = FString::Printf(TEXT("Found %d video files"), VideoFiles.Num());
}

void UBeatSyncWidget::StartAnalysis()
{
	if (!bHasValidAudio)
	{
		HandleError(TEXT("Please select an audio file first"));
		return;
	}

	UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem();
	if (!Subsystem)
	{
		HandleError(TEXT("BeatSync subsystem not available"));
		return;
	}

	bIsProcessing = true;
	CurrentProgress = 0.0f;
	StatusMessage = TEXT("Analyzing audio...");

	Subsystem->AnalyzeAudioFile(SelectedAudioFile);
}

void UBeatSyncWidget::StartProcessing()
{
	if (!bHasValidVideos)
	{
		HandleError(TEXT("Please select a folder with video files"));
		return;
	}

	if (SelectedOutputFolder.IsEmpty())
	{
		HandleError(TEXT("Please select an output folder"));
		return;
	}

	if (CurrentBeatData.BeatCount == 0)
	{
		HandleError(TEXT("Please analyze audio first"));
		return;
	}

	UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem();
	if (!Subsystem)
	{
		HandleError(TEXT("BeatSync subsystem not available"));
		return;
	}

	bIsProcessing = true;
	CurrentProgress = 0.0f;
	StatusMessage = TEXT("Processing videos...");

	Subsystem->ProcessVideos(VideoFiles, SelectedOutputFolder, ClipDuration);
}

void UBeatSyncWidget::CancelOperation()
{
	if (UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem())
	{
		Subsystem->CancelOperation();
		bIsProcessing = false;
		StatusMessage = TEXT("Operation cancelled");
	}
}

void UBeatSyncWidget::HandleAnalysisProgress(float Progress)
{
	CurrentProgress = Progress;
	StatusMessage = FString::Printf(TEXT("Analyzing: %.0f%%"), Progress * 100.0f);
}

void UBeatSyncWidget::HandleProcessingProgress(float Progress)
{
	CurrentProgress = Progress;
	StatusMessage = FString::Printf(TEXT("Processing: %.0f%%"), Progress * 100.0f);
}

void UBeatSyncWidget::HandleAnalysisComplete()
{
	bIsProcessing = false;
	CurrentProgress = 1.0f;

	if (UBeatsyncSubsystem* Subsystem = GetBeatsyncSubsystem())
	{
		CurrentBeatData = Subsystem->GetBeatData();
	}

	StatusMessage = FString::Printf(TEXT("Analysis complete: %d beats at %.1f BPM"), CurrentBeatData.BeatCount, CurrentBeatData.BPM);
	OnAnalysisCompleteUI.Broadcast();
}

void UBeatSyncWidget::HandleProcessingComplete()
{
	bIsProcessing = false;
	CurrentProgress = 1.0f;
	StatusMessage = TEXT("Processing complete!");
	OnProcessingCompleteUI.Broadcast();
}

void UBeatSyncWidget::HandleError(const FString& ErrorMessage)
{
	bIsProcessing = false;
	StatusMessage = FString::Printf(TEXT("Error: %s"), *ErrorMessage);
	OnErrorUI.Broadcast(ErrorMessage);
}
