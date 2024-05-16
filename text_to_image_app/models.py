from django.db import models

class GeneratedImage(models.Model):
    image = models.ImageField(upload_to='generated_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"GeneratedImage-{self.id}"
