from django.db import models

class VideoUpload(models.Model):
    video_url = models.URLField()
    processed_video = models.FileField(upload_to='processed_videos/', blank=True, null=True)
    result = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.video_url

