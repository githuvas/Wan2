# 🎬 ThreeStudio AI Movie Maker - Usage Guide

## 🚀 Quick Start

### 1. System Requirements
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional but recommended)
- 10GB+ free disk space

### 2. Installation
```bash
# Clone or navigate to the project directory
cd /workspace

# Create virtual environment
python3 -m venv threestudio_env

# Activate virtual environment
source threestudio_env/bin/activate

# Install dependencies
pip install -r requirements_minimal.txt
```

### 3. Verify Installation
```bash
# Run the test script to verify everything works
python test_simple.py
```

You should see:
```
🎬 ThreeStudio AI Movie Maker - System Test
==================================================
Testing imports...
✓ PyTorch: 2.8.0+cu128
✓ CUDA available: False
✓ NumPy: 2.2.6
✓ OpenCV: 4.12.0
✓ Gradio: 5.42.0
✓ Diffusers
✓ Transformers

Testing movie maker components...
✓ Movie maker imports successful
✓ Physics engine working
✓ Renderer working (frame shape: (480, 640, 3))

Testing configuration...
✓ Configuration created successfully

Testing character creation...
✓ Character created successfully

==================================================
🎉 All tests passed! The system is ready to use.
```

## 🎯 Usage Modes

### Web Interface (Recommended)
```bash
python threestudio_movie_maker_simple.py --mode web
```

This will launch a Gradio web interface at `http://localhost:7860` with:
- Text prompt input for movie scripts
- Duration slider (1-600 seconds)
- Quality settings (low/medium/high/ultra)
- Character setup (up to 5 characters with names and images)
- Output format selection (mp4/avi/mov)
- Real-time movie generation
- Download options

### Command Line Interface
```bash
python threestudio_movie_maker_simple.py --mode cli --prompt "Your movie script here" --duration 10 --output "my_movie.mp4"
```

## 🎭 Features

### 1. Character Consistency
- **Image Input**: Upload 1-10 character reference images
- **Consistency Engine**: Uses CLIP models to maintain character appearance
- **Personality**: Define character traits and behaviors
- **Voice Styles**: Different voice characteristics for each character

### 2. Physics Simulation
- **Real Physics**: Gravity, collision detection, forces
- **Rigid Bodies**: Objects with mass and momentum
- **Soft Bodies**: Deformable objects
- **Constraints**: Joints, springs, and connections

### 3. AI-Powered Generation
- **Text-to-Script**: Converts prompts into detailed movie scripts
- **Character Dialogue**: AI-generated conversations
- **Scene Description**: Detailed scene layouts and actions
- **Motion Planning**: Character movement and interactions

### 4. Rendering & Output
- **3D Rendering**: Simple but effective 3D graphics
- **Multiple Formats**: MP4, AVI, MOV support
- **Quality Presets**: Low to ultra quality options
- **Download Ready**: Direct download from web interface

## 📝 Example Prompts

### Action Scene
```
A brave knight in shining armor fights a massive dragon in a medieval castle. 
The knight wields a magical sword that glows with blue energy. The dragon breathes 
fire and the knight dodges while trying to find a weak spot. The castle has 
towering walls and flying banners.
```

### Sci-Fi Scene
```
Two advanced robots play chess in a futuristic city. The robots have sleek 
metallic bodies with glowing blue eyes. The city has flying cars, neon lights, 
and holographic advertisements everywhere. The chess pieces are floating 
holograms that move with smooth animations.
```

### Fantasy Scene
```
A young wizard learns magic in an enchanted forest. The wizard wears a blue 
robe and carries a wooden staff. The forest is filled with glowing mushrooms, 
talking animals, and floating magical orbs. The wizard practices casting spells 
that create colorful light effects.
```

## ⚙️ Advanced Configuration

### Custom Character Setup
```python
from threestudio_movie_maker_simple import CharacterConfig, MovieConfig, SimpleMovieGenerator

# Create character
character = CharacterConfig(
    name="HeroKnight",
    appearance_prompt="A brave knight in shining armor with a red cape",
    personality="brave and noble, always protects the innocent",
    voice_style="deep and confident",
    motion_style="graceful and powerful",
    physics_enabled=True,
    consistency_strength=0.8
)

# Create movie config
config = MovieConfig(
    title="Epic Adventure",
    duration_seconds=30.0,
    fps=30,
    resolution=(1920, 1080),
    output_format="mp4",
    quality_preset="high"
)

# Generate movie
generator = SimpleMovieGenerator(config)
generator.add_character(character)
output_path = generator.generate_movie("Your story here")
```

### Physics Customization
```python
from threestudio_movie_maker_simple import SimplePhysicsEngine

# Create physics engine with custom gravity
physics = SimplePhysicsEngine(gravity=(0.0, -5.0, 0.0))  # Reduced gravity

# Add objects
physics.add_object("hero", [0, 0, 0], [0, 0, 0], 1.0, 0.5)
physics.add_object("enemy", [5, 0, 0], [0, 0, 0], 2.0, 0.8)

# Apply forces
physics.apply_force("hero", [10, 0, 0])  # Push hero right
physics.step()  # Simulate physics
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_minimal.txt --force-reinstall
   ```

2. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install CPU-only version if needed
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory Issues**
   - Reduce resolution in MovieConfig
   - Use lower quality presets
   - Close other applications

4. **Slow Generation**
   - Use shorter durations for testing
   - Lower quality settings
   - Ensure GPU is being used (if available)

### Performance Tips

1. **For Fast Testing**
   ```python
   config = MovieConfig(
       duration_seconds=5.0,  # Short duration
       resolution=(640, 480),  # Lower resolution
       quality_preset="low"    # Lower quality
   )
   ```

2. **For High Quality**
   ```python
   config = MovieConfig(
       duration_seconds=60.0,   # Longer duration
       resolution=(1920, 1080), # Full HD
       quality_preset="ultra"   # Highest quality
   )
   ```

## 📁 File Structure

```
/workspace/
├── threestudio_movie_maker_simple.py  # Main application
├── test_simple.py                     # System test script
├── requirements_minimal.txt           # Dependencies
├── README.md                          # Project documentation
├── USAGE_GUIDE.md                     # This guide
├── output/                            # Generated movies
└── threestudio_env/                   # Virtual environment
```

## 🎉 Success Indicators

When everything is working correctly, you should see:

1. **Test Script**: All tests pass with green checkmarks
2. **Web Interface**: Gradio interface loads without errors
3. **Movie Generation**: Progress bars and status updates
4. **Output Files**: MP4 files created in the output directory

## 🆘 Getting Help

If you encounter issues:

1. Run `python test_simple.py` to verify system status
2. Check the console output for error messages
3. Ensure all dependencies are installed correctly
4. Verify you have sufficient disk space and memory

## 🚀 Next Steps

Once you're comfortable with the basic system:

1. **Experiment with different prompts** to see how the AI interprets them
2. **Try different character combinations** to test consistency
3. **Adjust physics parameters** for different effects
4. **Explore the code** to understand how it works
5. **Customize the system** for your specific needs

---

**🎬 Happy Movie Making!** 🎬

The ThreeStudio AI Movie Maker is now ready to create amazing 3D movies with AI characters, physics simulation, and realistic rendering - all from simple text prompts!