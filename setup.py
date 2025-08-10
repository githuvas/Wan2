#!/usr/bin/env python3
"""
Setup script for ThreeStudio AI Movie Maker
Installs all dependencies and sets up the environment
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreeStudioSetup:
    """Setup class for ThreeStudio AI Movie Maker"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements_simple.txt"
        self.output_dir = self.project_root / "output"
        self.models_dir = self.project_root / "models"
        self.assets_dir = self.project_root / "assets"
        
        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.assets_dir.mkdir(exist_ok=True)
    
    def check_system_requirements(self):
        """Check if system meets requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                logger.warning("CUDA not available. GPU acceleration will not be used.")
        except ImportError:
            logger.warning("PyTorch not installed yet.")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            logger.info(f"Available memory: {memory.total / (1024**3):.1f} GB")
            
            if memory.total < 8 * 1024**3:  # Less than 8GB
                logger.warning("Less than 8GB RAM detected. Performance may be limited.")
        except ImportError:
            logger.info("psutil not available, skipping memory check.")
        
        return True
    
    def install_dependencies(self):
        """Install all required dependencies"""
        logger.info("Installing dependencies...")
        
        # Upgrade pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to upgrade pip: {e}")
        
        # Install requirements
        if self.requirements_file.exists():
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
                ], check=True)
                logger.info("Dependencies installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                return False
        else:
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        return True
    
    def download_models(self):
        """Download required AI models"""
        logger.info("Downloading AI models...")
        
        models_to_download = [
            {
                "name": "stable-diffusion-v1-5",
                "type": "diffusers",
                "url": "runwayml/stable-diffusion-v1-5"
            },
            {
                "name": "clip-vit-base-patch32",
                "type": "transformers",
                "url": "openai/clip-vit-base-patch32"
            },
            {
                "name": "all-MiniLM-L6-v2",
                "type": "sentence-transformers",
                "url": "all-MiniLM-L6-v2"
            }
        ]
        
        for model in models_to_download:
            try:
                logger.info(f"Downloading {model['name']}...")
                
                if model['type'] == 'diffusers':
                    self._download_diffusers_model(model['url'], model['name'])
                elif model['type'] == 'transformers':
                    self._download_transformers_model(model['url'], model['name'])
                elif model['type'] == 'sentence-transformers':
                    self._download_sentence_transformer(model['url'], model['name'])
                
            except Exception as e:
                logger.error(f"Failed to download {model['name']}: {e}")
    
    def _download_diffusers_model(self, model_url: str, model_name: str):
        """Download a diffusers model"""
        try:
            from diffusers import DiffusionPipeline
            
            # This will download the model on first use
            logger.info(f"Pre-downloading {model_name}...")
            pipeline = DiffusionPipeline.from_pretrained(model_url)
            logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not pre-download {model_name}: {e}")
            logger.info(f"Model will be downloaded on first use")
    
    def _download_transformers_model(self, model_url: str, model_name: str):
        """Download a transformers model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info(f"Pre-downloading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_url)
            model = AutoModel.from_pretrained(model_url)
            logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not pre-download {model_name}: {e}")
            logger.info(f"Model will be downloaded on first use")
    
    def _download_sentence_transformer(self, model_url: str, model_name: str):
        """Download a sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Pre-downloading {model_name}...")
            model = SentenceTransformer(model_url)
            logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not pre-download {model_name}: {e}")
            logger.info(f"Model will be downloaded on first use")
    
    def create_config_file(self):
        """Create default configuration file"""
        config = {
            "system": {
                "max_memory_gb": 8,
                "use_gpu": True,
                "precision": "float16",
                "num_workers": 4
            },
            "rendering": {
                "default_resolution": [1920, 1080],
                "default_fps": 30,
                "render_quality": "high",
                "enable_ray_tracing": False
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
                "physics_aware_actions": True
            },
            "output": {
                "default_format": "mp4",
                "compression_quality": "high",
                "enable_audio": True
            }
        }
        
        config_file = self.project_root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration file created: {config_file}")
    
    def create_sample_assets(self):
        """Create sample assets for testing"""
        logger.info("Creating sample assets...")
        
        # Create sample character images
        sample_images_dir = self.assets_dir / "sample_characters"
        sample_images_dir.mkdir(exist_ok=True)
        
        # Create a simple test image
        try:
            from PIL import Image, ImageDraw
            
            # Create sample character images
            characters = ["Alice", "Bob", "Charlie"]
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
            
            for i, (name, color) in enumerate(zip(characters, colors)):
                img = Image.new('RGB', (256, 256), color)
                draw = ImageDraw.Draw(img)
                
                # Draw a simple character representation
                draw.ellipse([64, 64, 192, 192], fill=(255, 255, 255))
                draw.text((128, 128), name, fill=(0, 0, 0), anchor="mm")
                
                img_path = sample_images_dir / f"{name.lower()}.png"
                img.save(img_path)
                logger.info(f"Created sample character image: {img_path}")
                
        except ImportError:
            logger.warning("PIL not available, skipping sample image creation")
    
    def run_tests(self):
        """Run basic tests to ensure everything works"""
        logger.info("Running basic tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test imports
        test_modules = [
            "torch",
            "numpy",
            "cv2",
            "PIL",
            "gradio",
            "diffusers",
            "transformers",
            "open3d",
            "trimesh",
            "pyrender"
        ]
        
        for module in test_modules:
            total_tests += 1
            try:
                __import__(module)
                logger.info(f"✓ {module} imported successfully")
                tests_passed += 1
            except ImportError as e:
                logger.error(f"✗ Failed to import {module}: {e}")
        
        # Test GPU availability
        total_tests += 1
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("✓ CUDA available")
                tests_passed += 1
            else:
                logger.warning("⚠ CUDA not available")
        except ImportError:
            logger.error("✗ PyTorch not available")
        
        # Test physics engine
        total_tests += 1
        try:
            from advanced_physics_engine import AdvancedPhysicsEngine
            engine = AdvancedPhysicsEngine()
            logger.info("✓ Physics engine initialized")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Physics engine test failed: {e}")
        
        # Test script generator
        total_tests += 1
        try:
            from ai_script_generator import AIScriptGenerator
            generator = AIScriptGenerator()
            logger.info("✓ Script generator initialized")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Script generator test failed: {e}")
        
        logger.info(f"Tests completed: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            logger.info("🎉 All tests passed! Setup is complete.")
            return True
        else:
            logger.warning("⚠ Some tests failed. Please check the errors above.")
            return False
    
    def create_launch_scripts(self):
        """Create launch scripts for different platforms"""
        logger.info("Creating launch scripts...")
        
        # Create Python launch script
        launch_script = self.project_root / "launch.py"
        with open(launch_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Launch script for ThreeStudio AI Movie Maker
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the movie maker"""
    try:
        from threestudio_movie_maker import main as movie_maker_main
        movie_maker_main()
    except ImportError as e:
        print(f"Error importing movie maker: {e}")
        print("Please run setup.py first to install dependencies")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching movie maker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')
        
        # Make executable on Unix-like systems
        if platform.system() != "Windows":
            os.chmod(launch_script, 0o755)
        
        # Create batch file for Windows
        if platform.system() == "Windows":
            batch_file = self.project_root / "launch.bat"
            with open(batch_file, 'w') as f:
                f.write('''@echo off
echo Starting ThreeStudio AI Movie Maker...
python launch.py
pause
''')
        
        # Create shell script for Unix-like systems
        else:
            shell_file = self.project_root / "launch.sh"
            with open(shell_file, 'w') as f:
                f.write('''#!/bin/bash
echo "Starting ThreeStudio AI Movie Maker..."
python3 launch.py
''')
            os.chmod(shell_file, 0o755)
        
        logger.info("Launch scripts created")
    
    def create_readme(self):
        """Create a comprehensive README file"""
        readme_content = '''# ThreeStudio AI Movie Maker

A comprehensive text-to-3D movie generation system with AI characters, physics simulation, and realistic rendering.

## Features

- **AI Character Generation**: Create consistent characters with personality and voice
- **Physics Simulation**: Real-time physics with rigid bodies, soft bodies, and collision detection
- **3D Rendering**: High-quality 3D rendering with PyRender and Open3D
- **Script Generation**: AI-powered script generation with character consistency
- **Web Interface**: User-friendly Gradio interface
- **Multiple Output Formats**: Support for MP4, AVI, MOV formats
- **Offline Operation**: Works completely offline with local AI models

## Quick Start

1. **Install Dependencies**:
   ```bash
   python setup.py
   ```

2. **Launch the Application**:
   ```bash
   python launch.py
   ```

3. **Use the Web Interface**:
   - Open your browser to `http://localhost:7860`
   - Enter your movie prompt
   - Upload character images (optional)
   - Set duration and quality
   - Click "Generate Movie"

## Command Line Usage

```bash
# Generate a movie from command line
python threestudio_movie_maker.py --mode cli --prompt "A brave knight fights a dragon" --duration 30

# Launch web interface
python threestudio_movie_maker.py --mode web --port 7860
```

## Configuration

Edit `config.json` to customize:
- System settings (memory, GPU usage)
- Rendering quality
- Physics parameters
- AI model settings

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB free space for models and assets

## Architecture

The system consists of several key components:

1. **MovieGenerator**: Main orchestrator for movie generation
2. **PhysicsEngine**: Real-time physics simulation
3. **CharacterConsistencyManager**: Ensures character consistency
4. **ThreeStudioRenderer**: 3D rendering engine
5. **AIScriptGenerator**: AI-powered script generation
6. **AIVoiceGenerator**: Text-to-speech for characters

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU mode
2. **Model Download Failures**: Check internet connection and try again
3. **Import Errors**: Run `python setup.py` to install dependencies
4. **Performance Issues**: Reduce quality settings or use smaller models

### Getting Help

- Check the logs for detailed error messages
- Ensure all dependencies are installed correctly
- Verify system requirements are met

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.
'''
        
        readme_file = self.project_root / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        logger.info("README file created")
    
    def setup(self):
        """Run complete setup process"""
        logger.info("Starting ThreeStudio AI Movie Maker setup...")
        
        # Check system requirements
        if not self.check_system_requirements():
            logger.error("System requirements not met")
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            logger.error("Failed to install dependencies")
            return False
        
        # Download models
        self.download_models()
        
        # Create configuration
        self.create_config_file()
        
        # Create sample assets
        self.create_sample_assets()
        
        # Create launch scripts
        self.create_launch_scripts()
        
        # Create README
        self.create_readme()
        
        # Run tests
        if not self.run_tests():
            logger.warning("Some tests failed, but setup completed")
        
        logger.info("🎉 Setup completed successfully!")
        logger.info("You can now run: python launch.py")
        
        return True

def main():
    """Main setup function"""
    setup = ThreeStudioSetup()
    success = setup.setup()
    
    if success:
        print("\n" + "="*50)
        print("🎬 ThreeStudio AI Movie Maker Setup Complete!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run: python launch.py")
        print("2. Open your browser to: http://localhost:7860")
        print("3. Start creating amazing AI movies!")
        print("\nFor help, see README.md")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()