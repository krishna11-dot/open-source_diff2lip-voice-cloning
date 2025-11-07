# Voice Cloning & Lip Sync System

![Python](https://img.shields.io/badge/python-3.10-blue)
![Google Colab](https://img.shields.io/badge/platform-Google%20Colab-orange)
![Status](https://img.shields.io/badge/status-proof%20of%20concept-yellow)

An open-source proof-of-concept system that creates realistic talking videos from static images using Qwen based pretrained TTS model for voice cloning and diffusion-based lip synchronization.

## 🎬 Demo

**Live Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1Mh3rP1MsyisBNJxz50RqPpS9XGD-H1v6?usp=sharing)

**Sample Output**: [View Final Video](https://drive.google.com/file/d/1nxbVdTCzE7IbdeUJYsSRPbv4E1w_suMX/view?usp=sharing)

### Sample Script Used:
> "Hello and welcome. My name is Krishna and today, I'm going to demonstrate an incredible AI technology that brings static images to life. This system combines state-of-the-art voice cloning with advanced lip synchronization to create realistic talking videos from just a single photograph. The technology uses diffusion models and neural text-to-speech systems to generate natural looking results."

## 📋 What This Does

**Input:**
- Text script (manual input or uploaded file)
- Reference audio sample (for voice cloning)
- Front-facing portrait image

**Output:**
- MP4 video with lip-synced speech matching the cloned voice

## 🏗️ Architecture

### Three-Stage Pipeline:

```
1. Voice Synthesis (NeuTTS Air)
   Text + Reference Audio → Cloned Voice Audio
   
2. Video Preparation
   Static Image → 25 FPS Video (frame duplication)
   
3. Lip Synchronization (Diff2Lip)
   Video + Audio → Lip-Synced Final Video
```

### Technical Components:

**Stage 1: Voice Cloning**
- Model: NeuTTS Air (Qwen-based O.5B architecture)
- Size: 0.5B parameters (quantized to 4-bit)
- Type: Zero-shot voice cloning
- Performance: ~6 seconds for 5-second audio

**Stage 2: Image to Video**
- Converts static image to video format
- Frame rate: 25 FPS
- Duplicates frames to match audio duration
- Adds buffer frames to prevent audio overflow

**Stage 3: Lip Synchronization**
- Model: Diff2Lip (diffusion-based)
- Process:
  - Extracts facial landmarks
  - Detects and tracks mouth region
  - Generates lip movements per frame
  - Applies diffusion denoising (1000 steps)
  - Preserves facial identity
  - Blends modified mouth back to original face

## 🚀 Getting Started

### Prerequisites

- Google Colab account (recommended)
- Google Drive (for model checkpoints)

### Setup Instructions

1. **Open the Colab Notebook**
   ```
   https://colab.research.google.com/drive/1Mh3rP1MsyisBNJxz50RqPpS9XGD-H1v6?usp=sharing
   ```

2. **Connect to Runtime**
   - Runtime → Change runtime type → GPU (L4 or T4)

3. **Run Cells Sequentially**
   - The notebook handles all installations automatically
   - Model checkpoints are accessed via Google Drive

### What Gets Installed

The notebook automatically clones and installs:
- Diff2Lip repository
- NeuTTS Air repository
- Audio processing libraries (ffmpeg, soundfile)
- Computer vision libraries (OpenCV)
- ML frameworks (PyTorch, audio libraries)
- Message passing interface (MPI) for parallel processing

### Model Checkpoints

Models are stored in Google Drive and accessed directly:
- NeuTTS Air (quantized 4-bit version)
- Diff2Lip checkpoints
- No manual download required when using the notebook

## 💻 Usage

### Basic Workflow in Notebook:

```python
# 1. Install dependencies and clone repositories
# (Handled by notebook cells)

# 2. Load sample voices
# Pre-included samples: Dave (male), female voice

# 3. Input your script
script_text = """
Your text here...
"""

# 4. Upload portrait image
# Front-facing images work best

# 5. Run voice synthesis
# Clones voice using reference audio

# 6. Convert image to video
# Creates 25 FPS video from static image

# 7. Perform lip synchronization
# Generates final talking video

# 8. Download output
# MP4 file ready for use
```

### Key Parameters You Can Modify:

**Diff2Lip Parameters:**
```python
diffusion_steps = 1000  # Range: 50-1000 (lower = faster, less quality)
denoising_strength = 0.75
num_channels = 64
attention_resolutions = [32, 16, 8]
batch_size = 10  # For parallel frame processing
```

**Video Parameters:**
```python
fps = 25  # Frame rate
buffer_frames = 3  # Prevents audio overflow
```

## ⚡ Performance

### Generation Times:

| Hardware | Total Processing Time |
|----------|----------------------|
| L4 GPU (Colab) | ~2-3 minutes |
| T4 GPU (Colab) | ~14 minutes |
| CPU only | ~20-30 minutes |

### Component Breakdown:
- Voice synthesis: ~6 seconds (for 5-second audio)
- Image to video: <1 second
- Lip sync: Varies by GPU (main bottleneck)

### Model Sizes:
- NeuTTS Air (quantized): ~500MB
- NeuTTS Air (full): ~2GB
- Diff2Lip checkpoints: Accessed from Google Drive

## ✅ Strengths

- Zero-shot voice cloning (no training required)
- CPU-compatible (via model quantization)
- High-quality voice synthesis
- Diffusion-based lip sync (superior to GAN methods)
- Complete end-to-end pipeline
- Google Colab ready

## ⚠️ Current Limitations

### Image Requirements:
- **Front-facing portraits only** - side angles not supported
- Clear facial features required
- Avoid complex facial hair/beards (affects lip sync accuracy)

### Quality Issues:
- Mouth region may appear slightly blurry
- Lip movements can look artificial at times
- Edge artifacts around modified mouth area

### Performance:
- Diffusion models are computationally intensive
- 1000 diffusion steps can be slow without GPU
- Not optimized for real-time processing

### Production Status:
- **This is a proof of concept**
- Not production-ready
- Requires further optimization for deployment

## 🔧 Troubleshooting

### Common Issues:

**Version Conflicts:**
- PyTorch version matters for Diff2Lip
- Use Python 3.10 (latest versions may conflict)
- The notebook handles version management

**Model Loading Issues:**
- Ensure Google Drive access is granted
- Model checkpoints must be in correct paths
- Clone repositories before loading models

**Audio Playback in Colab:**
- Sharing audio requires uploading to Google Drive
- Direct playback from cells may not work in screen sharing

**GPU Memory:**
- L4 GPU recommended
- T4 GPU works but slower
- Reduce batch_size if memory issues occur

## 🛣️ Future Improvements

Potential enhancements for production version:
- [ ] Support for side-angle images
- [ ] Reduce mouth region blurriness
- [ ] Add emotion control in voice synthesis
- [ ] Include head movements and body animation
- [ ] Optimize for faster processing
- [ ] Add real-time blink generation
- [ ] Improve beard/facial hair handling
- [ ] Create standalone Python package
- [ ] Build web interface
- [ ] Add batch processing support

## 📊 Technical Details

### Voice Cloning (NeuTTS Air):
- Architecture:  (Qwen based architecture)
- Release: ~1.5 months ago (as of presentation)
- Quantization: 32-bit → 4-bit (GGUF-style)
- Can be further optimized for edge deployment

### Lip Synchronization (Diff2Lip):
- Type: Diffusion model
- Advantage: Better quality than GAN-based Wave2Lip
- Process: Iterative denoising of mouth region
- Preserves: Facial identity and context
- Modifies: Only mouth/lip region

### Audio Processing:
- Mel spectrogram extraction
- Phoneme-level feature extraction
- Audio-visual alignment
- Voice characteristic embedding

## 🤝 Collaboration

This proof of concept was developed through collaboration with:
- **ANTI-SOLUTIONS** - Testing and feedback on lip sync models
- **Ajitpal Singh Brar** and team - Parallel development and technical discussions

Collaborative efforts included comparing multiple approaches:
- Voice synthesis: ChatterboxTTS, NeuTTS Air
- Lip sync: Wave2Lip, Diff2Lip, SadTalker

## 🙏 Acknowledgments

- **NeuTTS Air** - Voice synthesis model
- **Diff2Lip** - Diffusion-based lip synchronization
- **ANTI-SOLUTIONS** - Collaboration and testing
- Research community for open-source frameworks

## 📄 License

This project is released as an open-source proof of concept.

## 📞 Contact

**Krishna Balachandran Nair**
- Email: krishnanair041@gmail.com
- LinkedIn: [krishna-balachandran-nair](https://www.linkedin.com/in/krishna-balachandran-nair-46621987/)
- GitHub: [@krishna11-dot](https://github.com/krishna11-dot)

## 🔬 Research Context

This project demonstrates the integration of:
- State-of-the-art voice cloning technology
- Diffusion-based video synthesis
- End-to-end multimodal AI pipelines

Developed as part of AI research and experimentation with latest generative models.

---

**Disclaimer**: This is a proof-of-concept research project. The output quality has limitations as documented above. For production applications, additional optimization and post-processing would be required.

**Note**: Audio samples are accesed by NeuTTS Air samples and model checkpoints are accessed via Google Drive in the notebook. The system does not include dataset creation or model training - it uses pre-trained models.
