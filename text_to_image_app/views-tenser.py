import tensorflow as tf
from django.shortcuts import render
from .forms import TextInputForm
from .models import GeneratedImage
import numpy as np
import random
from diffusers import DiffusionPipeline

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    tf.random.set_seed(seed)  # Set random seed
    
    # Convert the prompt to a TensorFlow tensor
    prompt_tensor = tf.constant([prompt])
    
    # Generate the image using the model
    image = pipe(prompt_tensor)
    
    # Convert the image tensor to a numpy array
    image_array = image.numpy()[0]
    
    return image_array

def home(request):
    if request.method == 'POST':
        form = TextInputForm(request.POST)
        if form.is_valid():
            prompt = form.cleaned_data['text']
            # Generate image using the prompt
            image = infer(prompt=prompt, negative_prompt="", seed=0, randomize_seed=False,
                          width=512, height=512, guidance_scale=0.0, num_inference_steps=2)
            # Save the generated image
            generated_image = GeneratedImage.objects.create(image=image)
            return render(request, 'text_to_image_app/home.html', {'form': form, 'generated_image': generated_image})
    else:
        form = TextInputForm()
    return render(request, 'text_to_image_app/home.html', {'form': form})
