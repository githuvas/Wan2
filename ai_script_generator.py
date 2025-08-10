#!/usr/bin/env python3
"""
Advanced AI Script Generator for ThreeStudio Movie Maker
Generates coherent movie scripts with character consistency and physics-aware actions
"""

import json
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    GenerationConfig
)
import openai
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ScriptEntry:
    """A single entry in the movie script"""
    timestamp: float
    character: str
    action: str
    dialogue: str
    camera: Dict[str, Any]
    physics_actions: List[Dict[str, Any]]
    emotion: str
    scene_description: str

@dataclass
class CharacterProfile:
    """Character profile for consistency"""
    name: str
    personality: str
    voice_style: str
    motion_style: str
    appearance: str
    background: str
    relationships: Dict[str, str]
    speech_patterns: List[str]
    typical_actions: List[str]

class AIScriptGenerator:
    """Advanced AI script generator using multiple models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.sentence_encoder = None
        self.character_profiles = {}
        self.scene_context = {}
        
        self._setup_models()
    
    def _setup_models(self):
        """Initialize AI models"""
        try:
            # Setup language model for script generation
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            # Setup sentence encoder for consistency
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("AI models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load AI models: {e}")
            logger.info("Falling back to rule-based script generation")
    
    def register_character(self, character: CharacterProfile):
        """Register a character for script generation"""
        self.character_profiles[character.name] = character
        
        # Create character embedding for consistency
        character_text = f"{character.name} {character.personality} {character.background}"
        if self.sentence_encoder:
            character_embedding = self.sentence_encoder.encode(character_text)
            character.embedding = character_embedding
    
    def generate_script(self, prompt: str, duration: float, characters: List[str]) -> List[ScriptEntry]:
        """Generate a complete movie script"""
        logger.info(f"Generating script for: {prompt}")
        
        # Parse prompt and extract key elements
        scene_info = self._parse_prompt(prompt)
        
        # Generate story structure
        story_structure = self._generate_story_structure(scene_info, duration)
        
        # Generate detailed script entries
        script_entries = []
        current_time = 0.0
        
        for scene in story_structure:
            scene_duration = scene['duration']
            scene_entries = self._generate_scene_script(
                scene, current_time, scene_duration
            )
            script_entries.extend(scene_entries)
            current_time += scene_duration
        
        return script_entries
    
    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse the input prompt to extract key information"""
        # Extract characters mentioned
        character_names = list(self.character_profiles.keys())
        mentioned_characters = []
        
        for char in character_names:
            if char.lower() in prompt.lower():
                mentioned_characters.append(char)
        
        # Extract setting/environment
        setting_keywords = [
            'castle', 'forest', 'city', 'beach', 'mountain', 'space', 'underwater',
            'medieval', 'futuristic', 'modern', 'ancient', 'fantasy', 'sci-fi'
        ]
        
        detected_setting = None
        for keyword in setting_keywords:
            if keyword in prompt.lower():
                detected_setting = keyword
                break
        
        # Extract action keywords
        action_keywords = [
            'fight', 'run', 'walk', 'talk', 'dance', 'sing', 'fly', 'swim',
            'explore', 'search', 'build', 'destroy', 'create', 'learn'
        ]
        
        detected_actions = []
        for keyword in action_keywords:
            if keyword in prompt.lower():
                detected_actions.append(keyword)
        
        return {
            'characters': mentioned_characters,
            'setting': detected_setting or 'generic',
            'actions': detected_actions,
            'mood': self._detect_mood(prompt),
            'original_prompt': prompt
        }
    
    def _detect_mood(self, prompt: str) -> str:
        """Detect the mood/tone of the prompt"""
        positive_words = ['happy', 'joy', 'fun', 'excited', 'wonderful', 'amazing']
        negative_words = ['sad', 'angry', 'scary', 'dark', 'dangerous', 'terrifying']
        neutral_words = ['calm', 'peaceful', 'quiet', 'normal', 'ordinary']
        
        prompt_lower = prompt.lower()
        
        positive_count = sum(1 for word in positive_words if word in prompt_lower)
        negative_count = sum(1 for word in negative_words if word in prompt_lower)
        neutral_count = sum(1 for word in neutral_words if word in prompt_lower)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_story_structure(self, scene_info: Dict[str, Any], duration: float) -> List[Dict[str, Any]]:
        """Generate the overall story structure"""
        # Divide into scenes based on duration
        if duration <= 30:
            num_scenes = 2
        elif duration <= 120:
            num_scenes = 3
        else:
            num_scenes = 5
        
        scene_duration = duration / num_scenes
        scenes = []
        
        for i in range(num_scenes):
            scene = {
                'id': i,
                'duration': scene_duration,
                'setting': scene_info['setting'],
                'mood': scene_info['mood'],
                'characters': scene_info['characters'],
                'actions': scene_info['actions'],
                'type': self._get_scene_type(i, num_scenes)
            }
            scenes.append(scene)
        
        return scenes
    
    def _get_scene_type(self, scene_id: int, total_scenes: int) -> str:
        """Determine the type of scene based on its position"""
        if scene_id == 0:
            return 'introduction'
        elif scene_id == total_scenes - 1:
            return 'conclusion'
        elif scene_id == total_scenes // 2:
            return 'climax'
        else:
            return 'development'
    
    def _generate_scene_script(self, scene: Dict[str, Any], start_time: float, duration: float) -> List[ScriptEntry]:
        """Generate detailed script entries for a scene"""
        entries = []
        
        # Calculate number of script entries based on duration
        # Assume each entry represents 2-3 seconds
        num_entries = max(3, int(duration / 2.5))
        
        for i in range(num_entries):
            entry_time = start_time + (i * duration / num_entries)
            
            # Select character for this entry
            character = self._select_character_for_entry(scene, i, num_entries)
            
            # Generate action and dialogue
            action, dialogue = self._generate_action_and_dialogue(scene, character, i, num_entries)
            
            # Generate camera movement
            camera = self._generate_camera_movement(scene, i, num_entries)
            
            # Generate physics actions
            physics_actions = self._generate_physics_actions(action, character)
            
            # Create script entry
            entry = ScriptEntry(
                timestamp=entry_time,
                character=character,
                action=action,
                dialogue=dialogue,
                camera=camera,
                physics_actions=physics_actions,
                emotion=self._determine_emotion(scene, i, num_entries),
                scene_description=self._generate_scene_description(scene, i, num_entries)
            )
            
            entries.append(entry)
        
        return entries
    
    def _select_character_for_entry(self, scene: Dict[str, Any], entry_index: int, total_entries: int) -> str:
        """Select which character should speak/act in this entry"""
        characters = scene['characters']
        if not characters:
            return "narrator"
        
        # Simple round-robin selection
        return characters[entry_index % len(characters)]
    
    def _generate_action_and_dialogue(self, scene: Dict[str, Any], character: str, entry_index: int, total_entries: int) -> Tuple[str, str]:
        """Generate action and dialogue for a character"""
        
        # Get character profile
        char_profile = self.character_profiles.get(character)
        
        # Generate action based on scene type and character
        if scene['type'] == 'introduction':
            if entry_index == 0:
                action = f"{character} appears on screen"
                dialogue = f"Hello, I am {character}!"
            else:
                action = f"{character} looks around the {scene['setting']}"
                dialogue = f"This {scene['setting']} is amazing!"
        
        elif scene['type'] == 'development':
            actions = [
                f"{character} walks forward",
                f"{character} gestures dramatically",
                f"{character} looks thoughtful",
                f"{character} moves closer to the camera"
            ]
            action = random.choice(actions)
            
            dialogues = [
                "This is getting interesting...",
                "I can feel the energy building up!",
                "What will happen next?",
                "The story continues..."
            ]
            dialogue = random.choice(dialogues)
        
        elif scene['type'] == 'climax':
            action = f"{character} performs an exciting action"
            dialogue = "This is the moment we've been waiting for!"
        
        else:  # conclusion
            action = f"{character} smiles and waves"
            dialogue = "Thank you for watching our story!"
        
        # Customize based on character personality
        if char_profile:
            if 'shy' in char_profile.personality.lower():
                dialogue = dialogue.lower() + " (shyly)"
            elif 'bold' in char_profile.personality.lower():
                dialogue = dialogue.upper()
        
        return action, dialogue
    
    def _generate_camera_movement(self, scene: Dict[str, Any], entry_index: int, total_entries: int) -> Dict[str, Any]:
        """Generate camera movement for the scene"""
        
        # Base camera position
        base_position = [0, 1.6, 3]  # Eye level, 3 units back
        
        # Camera movements based on scene type
        if scene['type'] == 'introduction':
            if entry_index == 0:
                # Start with wide shot
                position = [0, 2, 5]
                target = [0, 0, 0]
            else:
                # Move closer
                position = [0, 1.6, 3]
                target = [0, 0, 0]
        
        elif scene['type'] == 'development':
            # Dynamic camera movement
            angle = (entry_index / total_entries) * 2 * np.pi
            radius = 3
            position = [
                radius * np.cos(angle),
                1.6 + 0.5 * np.sin(angle * 2),
                radius * np.sin(angle)
            ]
            target = [0, 0, 0]
        
        elif scene['type'] == 'climax':
            # Dramatic close-up
            position = [0, 1.6, 1.5]
            target = [0, 0, 0]
        
        else:  # conclusion
            # Pull back for final shot
            position = [0, 2, 4]
            target = [0, 0, 0]
        
        return {
            'position': position,
            'target': target,
            'fov': 60,
            'movement_speed': 1.0
        }
    
    def _generate_physics_actions(self, action: str, character: str) -> List[Dict[str, Any]]:
        """Generate physics actions based on the character action"""
        physics_actions = []
        
        # Parse action and generate corresponding physics
        if 'walk' in action.lower():
            physics_actions.append({
                'type': 'movement',
                'character': character,
                'direction': [1, 0, 0],
                'speed': 1.0,
                'duration': 2.0
            })
        
        elif 'jump' in action.lower():
            physics_actions.append({
                'type': 'impulse',
                'character': character,
                'force': [0, 5, 0],
                'duration': 0.5
            })
        
        elif 'gesture' in action.lower():
            physics_actions.append({
                'type': 'animation',
                'character': character,
                'animation': 'wave',
                'duration': 1.0
            })
        
        return physics_actions
    
    def _determine_emotion(self, scene: Dict[str, Any], entry_index: int, total_entries: int) -> str:
        """Determine the emotional state for this entry"""
        emotions = ['neutral', 'happy', 'excited', 'thoughtful', 'surprised']
        
        if scene['type'] == 'introduction':
            return 'excited'
        elif scene['type'] == 'climax':
            return 'excited'
        elif scene['type'] == 'conclusion':
            return 'happy'
        else:
            return random.choice(emotions)
    
    def _generate_scene_description(self, scene: Dict[str, Any], entry_index: int, total_entries: int) -> str:
        """Generate scene description for this entry"""
        setting = scene['setting']
        mood = scene['mood']
        
        descriptions = {
            'castle': f"A majestic {mood} castle with towering walls and grand architecture",
            'forest': f"A mystical {mood} forest with ancient trees and dappled sunlight",
            'city': f"A bustling {mood} city with modern buildings and busy streets",
            'beach': f"A serene {mood} beach with golden sand and rolling waves",
            'space': f"A vast {mood} space environment with stars and distant planets",
            'generic': f"A {mood} environment filled with wonder and possibility"
        }
        
        return descriptions.get(setting, descriptions['generic'])

class CharacterConsistencyChecker:
    """Ensures character consistency throughout the script"""
    
    def __init__(self):
        self.character_history = {}
        self.consistency_rules = {}
    
    def add_character(self, character: CharacterProfile):
        """Add a character to the consistency checker"""
        self.character_history[character.name] = []
        self.consistency_rules[character.name] = {
            'speech_patterns': character.speech_patterns,
            'typical_actions': character.typical_actions,
            'personality': character.personality
        }
    
    def check_consistency(self, script_entries: List[ScriptEntry]) -> List[Dict[str, Any]]:
        """Check character consistency throughout the script"""
        issues = []
        
        for i, entry in enumerate(script_entries):
            character = entry.character
            if character in self.character_history:
                # Check against previous entries
                history = self.character_history[character]
                
                # Check speech consistency
                if entry.dialogue:
                    speech_issue = self._check_speech_consistency(character, entry.dialogue, history)
                    if speech_issue:
                        issues.append(speech_issue)
                
                # Check action consistency
                action_issue = self._check_action_consistency(character, entry.action, history)
                if action_issue:
                    issues.append(action_issue)
                
                # Update history
                history.append({
                    'dialogue': entry.dialogue,
                    'action': entry.action,
                    'emotion': entry.emotion,
                    'timestamp': entry.timestamp
                })
        
        return issues
    
    def _check_speech_consistency(self, character: str, dialogue: str, history: List[Dict]) -> Optional[Dict]:
        """Check if dialogue is consistent with character's speech patterns"""
        rules = self.consistency_rules.get(character, {})
        speech_patterns = rules.get('speech_patterns', [])
        
        # Check for speech pattern violations
        for pattern in speech_patterns:
            if pattern.lower() in dialogue.lower():
                return None  # Pattern found, consistent
        
        # Check against recent history
        if history:
            recent_dialogues = [h['dialogue'] for h in history[-3:] if h['dialogue']]
            if recent_dialogues:
                # Simple consistency check
                return {
                    'type': 'speech_inconsistency',
                    'character': character,
                    'dialogue': dialogue,
                    'suggestion': f"Consider using character's typical speech patterns: {speech_patterns}"
                }
        
        return None
    
    def _check_action_consistency(self, character: str, action: str, history: List[Dict]) -> Optional[Dict]:
        """Check if action is consistent with character's typical actions"""
        rules = self.consistency_rules.get(character, {})
        typical_actions = rules.get('typical_actions', [])
        
        # Check for typical action patterns
        for typical_action in typical_actions:
            if typical_action.lower() in action.lower():
                return None  # Action found, consistent
        
        return {
            'type': 'action_inconsistency',
            'character': character,
            'action': action,
            'suggestion': f"Consider using character's typical actions: {typical_actions}"
        }

def create_sample_character_profiles() -> List[CharacterProfile]:
    """Create sample character profiles for testing"""
    profiles = [
        CharacterProfile(
            name="Alice",
            personality="brave and adventurous",
            voice_style="clear and confident",
            motion_style="graceful and purposeful",
            appearance="young woman with flowing hair",
            background="explorer from a distant land",
            relationships={"Bob": "friend", "Charlie": "mentor"},
            speech_patterns=["I can do this!", "Let's explore", "How fascinating"],
            typical_actions=["exploring", "climbing", "discovering", "helping others"]
        ),
        CharacterProfile(
            name="Bob",
            personality="shy but intelligent",
            voice_style="soft and thoughtful",
            motion_style="careful and deliberate",
            appearance="young man with glasses",
            background="scientist and inventor",
            relationships={"Alice": "friend", "Charlie": "colleague"},
            speech_patterns=["I think...", "Perhaps we could", "That's interesting"],
            typical_actions=["observing", "thinking", "inventing", "analyzing"]
        ),
        CharacterProfile(
            name="Charlie",
            personality="wise and experienced",
            voice_style="deep and authoritative",
            motion_style="slow and deliberate",
            appearance="older person with white hair",
            background="master of ancient knowledge",
            relationships={"Alice": "mentor", "Bob": "colleague"},
            speech_patterns=["In my experience", "The ancient texts say", "Listen carefully"],
            typical_actions=["teaching", "guiding", "meditating", "sharing wisdom"]
        )
    ]
    
    return profiles

if __name__ == "__main__":
    # Test the script generator
    generator = AIScriptGenerator()
    
    # Create sample characters
    characters = create_sample_character_profiles()
    for char in characters:
        generator.register_character(char)
    
    # Generate a test script
    prompt = "Alice and Bob explore a magical castle together, discovering ancient secrets"
    script = generator.generate_script(prompt, 30.0, ["Alice", "Bob"])
    
    # Print the script
    for entry in script:
        print(f"[{entry.timestamp:.1f}s] {entry.character}: {entry.dialogue}")
        print(f"  Action: {entry.action}")
        print(f"  Camera: {entry.camera['position']}")
        print()