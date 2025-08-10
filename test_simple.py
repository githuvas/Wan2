#!/usr/bin/env python3
"""
Simple test script for ThreeStudio AI Movie Maker
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import movie maker components globally
try:
    from threestudio_movie_maker_simple import (
        SimpleMovieGenerator, 
        CharacterConfig, 
        MovieConfig,
        SimplePhysicsEngine,
        SimpleRenderer
    )
    MOVIE_MAKER_IMPORTED = True
except ImportError:
    MOVIE_MAKER_IMPORTED = False

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import gradio as gr
        print(f"✓ Gradio: {gr.__version__}")
    except ImportError as e:
        print(f"✗ Gradio import failed: {e}")
        return False
    
    try:
        from diffusers import DiffusionPipeline
        print("✓ Diffusers")
    except ImportError as e:
        print(f"✗ Diffusers import failed: {e}")
        return False
    
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    return True

def test_movie_maker():
    """Test the movie maker components"""
    print("\nTesting movie maker components...")
    
    if not MOVIE_MAKER_IMPORTED:
        print("✗ Movie maker imports failed")
        return False
    
    print("✓ Movie maker imports successful")
    
    try:
        # Test physics engine
        physics = SimplePhysicsEngine()
        physics.add_object("test", [0, 0, 0], [0, 0, 0], 1.0)
        physics.step()
        print("✓ Physics engine working")
    except Exception as e:
        print(f"✗ Physics engine failed: {e}")
        return False
    
    try:
        # Test renderer
        renderer = SimpleRenderer(640, 480)
        renderer.add_character("test", [0, 0, 0])
        frame = renderer.render_frame()
        print(f"✓ Renderer working (frame shape: {frame.shape})")
    except Exception as e:
        print(f"✗ Renderer failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        config = MovieConfig(
            title="Test Movie",
            duration_seconds=5.0,
            fps=30,
            resolution=(640, 480),
            output_format="mp4",
            quality_preset="medium"
        )
        print("✓ Configuration created successfully")
        print(f"  - Title: {config.title}")
        print(f"  - Duration: {config.duration_seconds}s")
        print(f"  - FPS: {config.fps}")
        print(f"  - Resolution: {config.resolution}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_character():
    """Test character creation"""
    print("\nTesting character creation...")
    
    try:
        character = CharacterConfig(
            name="TestKnight",
            appearance_prompt="A brave knight in shining armor",
            personality="brave and noble",
            voice_style="deep and confident",
            motion_style="graceful",
            physics_enabled=True
        )
        print("✓ Character created successfully")
        print(f"  - Name: {character.name}")
        print(f"  - Personality: {character.personality}")
        print(f"  - Physics enabled: {character.physics_enabled}")
        return True
    except Exception as e:
        print(f"✗ Character creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎬 ThreeStudio AI Movie Maker - System Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test movie maker components
    if not test_movie_maker():
        print("\n❌ Movie maker tests failed")
        return False
    
    # Test configuration
    if not test_config():
        print("\n❌ Configuration tests failed")
        return False
    
    # Test character creation
    if not test_character():
        print("\n❌ Character tests failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! The system is ready to use.")
    print("\nYou can now run:")
    print("  python threestudio_movie_maker_simple.py --mode web")
    print("  python threestudio_movie_maker_simple.py --mode cli --prompt 'Your story here'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)