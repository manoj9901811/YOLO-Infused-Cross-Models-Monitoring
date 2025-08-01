from django import forms

class VideoURLForm(forms.Form):
    video_url = forms.URLField(required=False, label='Video URL')
    video_file = forms.FileField(required=False, label='Upload Video')
    image_file = forms.ImageField(required=False, label='Upload Image')  # ✅ Add this line

    def clean(self):
        cleaned_data = super().clean()
        video_url = cleaned_data.get('video_url')
        video_file = cleaned_data.get('video_file')
        image_file = cleaned_data.get('image_file')

        if not video_url and not video_file and not image_file:
            raise forms.ValidationError("Please provide a video URL, upload a video, or upload an image.")
        return cleaned_data
