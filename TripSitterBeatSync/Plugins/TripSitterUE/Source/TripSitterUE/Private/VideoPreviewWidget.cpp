#include "VideoPreviewWidget.h"
#include "MediaPlayer.h"
#include "MediaTexture.h"
#include "FileMediaSource.h"

void UVideoPreviewWidget::NativeConstruct()
{
	Super::NativeConstruct();

	// Create media player
	MediaPlayer = NewObject<UMediaPlayer>(this);
	MediaPlayer->OnMediaOpened.AddDynamic(this, &UVideoPreviewWidget::HandleMediaOpened);
	MediaPlayer->OnMediaOpenFailed.AddDynamic(this, &UVideoPreviewWidget::HandleMediaOpenFailed);
	MediaPlayer->OnEndReached.AddDynamic(this, &UVideoPreviewWidget::HandleEndReached);

	// Create media texture
	MediaTexture = NewObject<UMediaTexture>(this);
	MediaTexture->SetMediaPlayer(MediaPlayer);
	MediaTexture->UpdateResource();
}

void UVideoPreviewWidget::NativeDestruct()
{
	if (MediaPlayer)
	{
		MediaPlayer->Close();
		MediaPlayer->OnMediaOpened.RemoveDynamic(this, &UVideoPreviewWidget::HandleMediaOpened);
		MediaPlayer->OnMediaOpenFailed.RemoveDynamic(this, &UVideoPreviewWidget::HandleMediaOpenFailed);
		MediaPlayer->OnEndReached.RemoveDynamic(this, &UVideoPreviewWidget::HandleEndReached);
	}

	Super::NativeDestruct();
}

void UVideoPreviewWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
	Super::NativeTick(MyGeometry, InDeltaTime);

	// Report playback time changes
	if (MediaPlayer && MediaPlayer->IsPlaying())
	{
		float CurrentTime = GetCurrentTime();
		if (FMath::Abs(CurrentTime - LastReportedTime) > 0.033f) // ~30fps update
		{
			LastReportedTime = CurrentTime;
			OnPlaybackTimeChanged.Broadcast(CurrentTime);
		}
	}
}

bool UVideoPreviewWidget::LoadVideo(const FString& FilePath)
{
	if (!MediaPlayer)
	{
		return false;
	}

	CurrentVideoPath = FilePath;
	bIsVideoLoaded = false;

	// Open the file directly
	return MediaPlayer->OpenFile(FilePath);
}

void UVideoPreviewWidget::Play()
{
	if (MediaPlayer && bIsVideoLoaded)
	{
		MediaPlayer->Play();
	}
}

void UVideoPreviewWidget::Pause()
{
	if (MediaPlayer)
	{
		MediaPlayer->Pause();
	}
}

void UVideoPreviewWidget::Stop()
{
	if (MediaPlayer)
	{
		MediaPlayer->Pause();
		MediaPlayer->Seek(FTimespan::Zero());
		LastReportedTime = 0.0f;
	}
}

void UVideoPreviewWidget::SeekToTime(float TimeInSeconds)
{
	if (MediaPlayer && bIsVideoLoaded)
	{
		FTimespan SeekTime = FTimespan::FromSeconds(TimeInSeconds);
		MediaPlayer->Seek(SeekTime);
	}
}

float UVideoPreviewWidget::GetCurrentTime() const
{
	if (MediaPlayer)
	{
		return static_cast<float>(MediaPlayer->GetTime().GetTotalSeconds());
	}
	return 0.0f;
}

float UVideoPreviewWidget::GetDuration() const
{
	if (MediaPlayer)
	{
		return static_cast<float>(MediaPlayer->GetDuration().GetTotalSeconds());
	}
	return 0.0f;
}

bool UVideoPreviewWidget::IsPlaying() const
{
	return MediaPlayer && MediaPlayer->IsPlaying();
}

void UVideoPreviewWidget::HandleMediaOpened(FString OpenedUrl)
{
	bIsVideoLoaded = true;
	OnVideoLoaded.Broadcast();
}

void UVideoPreviewWidget::HandleMediaOpenFailed(FString FailedUrl)
{
	bIsVideoLoaded = false;
	UE_LOG(LogTemp, Error, TEXT("Failed to open video: %s"), *FailedUrl);
}

void UVideoPreviewWidget::HandleEndReached()
{
	OnVideoEnded.Broadcast();
}
