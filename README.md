# 🎬 ThreeStudio AI Movie Maker

A comprehensive **text-to-3D movie generation system** with AI characters, physics simulation, and realistic rendering. Create amazing AI movies with consistent characters, real-time physics, and high-quality 3D graphics - all from simple text prompts!

## ✨ Features

### 🎭 AI Character Generation
- **Consistent Characters**: AI characters maintain personality and appearance throughout the movie
- **Multiple Characters**: Support for up to 10 characters with unique personalities
- **Character Images**: Upload reference images (1-10) for character consistency
- **Voice Generation**: AI-powered text-to-speech for each character
- **Personality Profiles**: Detailed character backgrounds, speech patterns, and behaviors

### 🌍 Physics World Engine
- **Real-time Physics**: Advanced physics simulation with rigid bodies, soft bodies, and fluids
- **Collision Detection**: Accurate collision detection and response
- **Constraints**: Distance, spring, hinge, and fixed constraints between objects
- **Materials**: Realistic materials (metal, wood, rubber, cloth, water, ice)
- **Gravity & Forces**: Customizable gravity, wind, and force application

### 🎨 3D Rendering & Visualization
- **High-Quality Rendering**: PyRender-based 3D rendering with realistic lighting
- **Multiple Formats**: Support for MP4, AVI, MOV output formats
- **Customizable Quality**: Low, medium, high, and ultra quality presets
- **Camera Control**: Dynamic camera movements and positioning
- **Lighting System**: Ambient, directional, and point lighting

### 🤖 AI-Powered Script Generation
- **Intelligent Scripts**: AI generates coherent movie scripts from text prompts
- **Character Consistency**: Maintains character personality and speech patterns
- **Scene Structure**: Automatic scene division and story progression
- **Physics-Aware Actions**: Scripts include physics-based character actions
- **Emotion Tracking**: Character emotions and reactions throughout the story

### 🎬 Movie Generation Pipeline
- **Duration Control**: 1 second to 10 minutes movie duration
- **Frame-by-Frame**: Real-time frame generation with physics integration
- **Audio Integration**: Synchronized audio with character dialogue
- **Download Options**: Direct download of generated videos
- **Batch Processing**: Generate multiple movies with different settings

### 🌐 Web Interface
- **Gradio UI**: Beautiful, modern web interface
- **Real-time Preview**: Live preview of generation progress
- **Character Management**: Easy character setup and configuration
- **Example Prompts**: Pre-built example scenarios
- **Progress Tracking**: Real-time progress indicators

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd threestudio-movie-maker

# Run the setup script
python setup.py
```

### 2. Launch the Application

```bash
# Launch the web interface
python launch.py

# Or use the main script
python threestudio_movie_maker.py --mode web
```

### 3. Create Your First Movie

1. Open your browser to `http://localhost:7860`
2. Enter a movie prompt: *"A brave knight fights a dragon in a medieval castle"*
3. Set duration (1-600 seconds)
4. Choose quality level
5. Upload character images (optional)
6. Click "Generate Movie" and wait for the magic!

## 📖 Detailed Usage

### Command Line Interface

```bash
# Generate a movie from command line
python threestudio_movie_maker.py --mode cli \
    --prompt "A brave knight fights a dragon" \
    --duration 30 \
    --output "my_movie.mp4"

# Launch web interface on custom port
python threestudio_movie_maker.py --mode web --port 8080

# Generate with specific quality
python threestudio_movie_maker.py --mode cli \
    --prompt "Two robots play chess" \
    --duration 20 \
    --quality ultra
```

### Python API

```python
from threestudio_movie_maker import MovieGenerator, CharacterConfig, MovieConfig

# Create movie configuration
config = MovieConfig(
    title="My AI Movie",
    duration_seconds=30.0,
    fps=30,
    resolution=(1920, 1080),
    output_format="mp4",
    quality_preset="high"
)

# Create movie generator
generator = MovieGenerator(config)

# Add characters
alice = CharacterConfig(
    name="Alice",
    appearance_prompt="A young woman with flowing hair",
    personality="brave and adventurous",
    voice_style="clear and confident",
    motion_style="graceful",
    physics_enabled=True
)
generator.add_character(alice)

# Generate movie
output_path = generator.generate_movie(
    "Alice explores a magical forest and discovers ancient secrets"
)
print(f"Movie generated: {output_path}")
```

### Advanced Configuration

Edit `config.json` to customize system settings:

```json
{
  "system": {
    "max_memory_gb": 8,
    "use_gpu": true,
    "precision": "float16",
    "num_workers": 4
  },
  "rendering": {
    "default_resolution": [1920, 1080],
    "default_fps": 30,
    "render_quality": "high",
    "enable_ray_tracing": false
  },
  "physics": {
    "gravity": [0.0, -9.81, 0.0],
    "substeps": 10,
    "max_velocity": 100.0,
    "air_resistance": 0.01
  },
  "ai": {
    "script_generation_model": "microsoft/DialoGPT-medium",
    "character_consistency_strength": 0.8,
    "physics_aware_actions": true
  }
}
```

## 🏗️ System Architecture

The ThreeStudio AI Movie Maker consists of several interconnected components:

### Core Components

1. **MovieGenerator** (`threestudio_movie_maker.py`)
   - Main orchestrator for movie generation
   - Coordinates all subsystems
   - Manages the complete pipeline

2. **AdvancedPhysicsEngine** (`advanced_physics_engine.py`)
   - Real-time physics simulation
   - Rigid body dynamics
   - Collision detection and response
   - Constraint solving

3. **AIScriptGenerator** (`ai_script_generator.py`)
   - AI-powered script generation
   - Character consistency management
   - Scene structure generation
   - Physics-aware action planning

4. **ThreeStudioRenderer** (integrated)
   - 3D rendering engine
   - PyRender integration
   - Camera and lighting management
   - Frame generation

5. **CharacterConsistencyManager** (integrated)
   - Character embedding management
   - Consistency checking
   - Personality maintenance
   - Reference image processing

### Data Flow

```
Text Prompt → AI Script Generator → Physics Engine → 3D Renderer → Video Output
     ↓              ↓                    ↓              ↓
Character Config → Consistency Check → Physics Actions → Frame Generation
```

## 🎯 Example Scenarios

### Adventure Movie
```
Prompt: "A brave knight fights a dragon in a medieval castle"
Duration: 30 seconds
Characters: Knight, Dragon
Features: Combat physics, dramatic camera angles, character dialogue
```

### Sci-Fi Story
```
Prompt: "Two robots play chess in a futuristic city with flying cars"
Duration: 45 seconds
Characters: Robot1, Robot2
Features: Futuristic environment, AI dialogue, smooth animations
```

### Fantasy Tale
```
Prompt: "A young wizard learns magic in an enchanted forest"
Duration: 60 seconds
Characters: Wizard, Forest Spirit
Features: Magical effects, nature physics, mystical atmosphere
```

## 🔧 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 10GB free space
- **GPU**: Any (CPU-only mode available)

### Recommended Requirements
- **Python**: 3.9 or higher
- **RAM**: 16GB or more
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher

### Supported Platforms
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)
- ✅ Windows 10/11
- ✅ macOS 10.15+

## 📦 Installation Details

### Automatic Installation
```bash
python setup.py
```

The setup script will:
- Check system requirements
- Install all dependencies
- Download AI models
- Create configuration files
- Generate sample assets
- Run system tests

### Manual Installation
```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Download models (optional, will download on first use)
python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

## 🎮 Demo and Testing

### Run Comprehensive Demo
```bash
python demo.py --demo all
```

### Run Specific Demos
```bash
# Physics engine demo
python demo.py --demo physics

# Script generation demo
python demo.py --demo script

# Character consistency demo
python demo.py --demo consistency

# Movie generation demo
python demo.py --demo movie --movie-type action
```

### Demo Features
- **Physics Simulation**: Bouncing balls, pendulums, constraints
- **Script Generation**: Multiple scenarios with character dialogue
- **Consistency Checking**: Character personality validation
- **Movie Generation**: Complete pipeline demonstration

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in config.json
"system": {
    "max_memory_gb": 4,
    "use_gpu": false
}
```

#### Model Download Failures
```bash
# Check internet connection
# Models will download on first use
# Use --offline flag for offline mode
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements_enhanced.txt --force-reinstall
```

#### Performance Issues
```bash
# Reduce quality settings
# Use CPU mode if GPU is slow
# Increase system memory
```

### Getting Help

1. **Check Logs**: Look for detailed error messages in console output
2. **Verify Dependencies**: Ensure all packages are installed correctly
3. **System Requirements**: Confirm your system meets minimum requirements
4. **Configuration**: Check `config.json` for proper settings

## 🔬 Advanced Features

### Custom Physics Materials
```python
from advanced_physics_engine import PhysicsMaterial

custom_material = PhysicsMaterial(
    name="magical_crystal",
    density=2500.0,
    friction=0.1,
    restitution=0.9,
    damping=0.05,
    color=(0.8, 0.2, 1.0)
)
```

### Custom Character Profiles
```python
from ai_script_generator import CharacterProfile

wizard = CharacterProfile(
    name="Gandalf",
    personality="wise and mysterious",
    voice_style="deep and authoritative",
    motion_style="deliberate and graceful",
    appearance="old wizard with long white beard",
    background="ancient guardian of Middle-earth",
    relationships={"Frodo": "mentor", "Saruman": "rival"},
    speech_patterns=["You shall not pass!", "A wizard is never late"],
    typical_actions=["casting spells", "smoking pipe", "giving advice"]
)
```

### Custom Rendering Settings
```python
from threestudio_movie_maker import ThreeStudioRenderer

renderer = ThreeStudioRenderer(
    width=3840,  # 4K resolution
    height=2160,
    enable_ray_tracing=True,
    shadow_quality="ultra"
)
```

## 📊 Performance Benchmarks

### Generation Times (RTX 4090)
- **10-second movie**: ~2-3 minutes
- **30-second movie**: ~5-7 minutes
- **60-second movie**: ~10-15 minutes

### Memory Usage
- **Low quality**: 4-6GB RAM
- **Medium quality**: 6-8GB RAM
- **High quality**: 8-12GB RAM
- **Ultra quality**: 12-16GB RAM

### GPU Requirements
- **Minimum**: 4GB VRAM
- **Recommended**: 8GB+ VRAM
- **Optimal**: 12GB+ VRAM

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd threestudio-movie-maker

# Install development dependencies
pip install -r requirements_enhanced.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Type checking
mypy .
```

### Areas for Contribution
- **Physics Engine**: Add new physics features
- **AI Models**: Integrate new AI models
- **Rendering**: Improve 3D rendering quality
- **UI/UX**: Enhance web interface
- **Documentation**: Improve docs and examples
- **Testing**: Add more test cases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Diffusers**: Diffusion model pipeline
- **PyRender**: 3D rendering engine
- **Gradio**: Web interface framework
- **Open3D**: 3D data processing
- **Transformers**: AI model library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

## 🎉 What's Next?

The ThreeStudio AI Movie Maker is constantly evolving. Upcoming features:

- **Multi-language Support**: Generate movies in multiple languages
- **Advanced AI Models**: Integration with latest AI models
- **Real-time Collaboration**: Multi-user movie creation
- **VR/AR Support**: Virtual and augmented reality output
- **Cloud Rendering**: Distributed rendering for faster generation
- **Animation Library**: Pre-built character animations
- **Sound Effects**: AI-generated sound effects and music

---

**🎬 Start creating amazing AI movies today!**

*Built with ❤️ using cutting-edge AI and computer graphics technology.*
