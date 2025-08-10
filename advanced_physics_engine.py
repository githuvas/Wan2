#!/usr/bin/env python3
"""
Advanced Physics Engine for ThreeStudio Movie Maker
Realistic physics simulation with rigid bodies, soft bodies, and collision detection
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PhysicsObjectType(Enum):
    """Types of physics objects"""
    RIGID_BODY = "rigid_body"
    SOFT_BODY = "soft_body"
    PARTICLE = "particle"
    FLUID = "fluid"
    CLOTH = "cloth"

class CollisionShape(Enum):
    """Types of collision shapes"""
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    MESH = "mesh"

@dataclass
class PhysicsMaterial:
    """Physics material properties"""
    name: str
    density: float = 1000.0  # kg/m³
    friction: float = 0.5
    restitution: float = 0.3  # bounciness
    damping: float = 0.1
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)

@dataclass
class PhysicsObject:
    """Base physics object"""
    id: str
    object_type: PhysicsObjectType
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray
    mass: float
    material: PhysicsMaterial
    collision_shape: CollisionShape
    collision_radius: float = 0.1
    collision_size: np.ndarray = None
    active: bool = True
    forces: np.ndarray = None
    torques: np.ndarray = None
    
    def __post_init__(self):
        if self.forces is None:
            self.forces = np.zeros(3)
        if self.torques is None:
            self.torques = np.zeros(3)
        if self.collision_size is None:
            self.collision_size = np.array([0.1, 0.1, 0.1])

@dataclass
class Constraint:
    """Physics constraint between objects"""
    id: str
    object1_id: str
    object2_id: str
    constraint_type: str  # "distance", "hinge", "spring", "fixed"
    parameters: Dict[str, Any]

class AdvancedPhysicsEngine:
    """Advanced physics engine with realistic simulations"""
    
    def __init__(self, 
                 gravity: np.ndarray = np.array([0.0, -9.81, 0.0]),
                 air_resistance: float = 0.01,
                 max_velocity: float = 100.0,
                 substeps: int = 10):
        
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.max_velocity = max_velocity
        self.substeps = substeps
        self.time_step = 1.0 / 60.0
        
        # Physics objects and constraints
        self.objects: Dict[str, PhysicsObject] = {}
        self.constraints: List[Constraint] = []
        
        # Collision detection
        self.broad_phase_grid = {}
        self.grid_size = 1.0
        
        # Materials
        self.materials = self._create_default_materials()
        
        # Performance tracking
        self.frame_count = 0
        self.collision_checks = 0
        
        logger.info("Advanced Physics Engine initialized")
    
    def _create_default_materials(self) -> Dict[str, PhysicsMaterial]:
        """Create default physics materials"""
        return {
            "default": PhysicsMaterial("default"),
            "metal": PhysicsMaterial("metal", density=7800, friction=0.3, restitution=0.1),
            "wood": PhysicsMaterial("wood", density=600, friction=0.6, restitution=0.2),
            "rubber": PhysicsMaterial("rubber", density=1200, friction=0.8, restitution=0.8),
            "ice": PhysicsMaterial("ice", density=920, friction=0.1, restitution=0.1),
            "cloth": PhysicsMaterial("cloth", density=200, friction=0.4, restitution=0.0),
            "water": PhysicsMaterial("water", density=1000, friction=0.0, restitution=0.0)
        }
    
    def add_object(self, 
                   obj_id: str,
                   object_type: PhysicsObjectType,
                   position: np.ndarray,
                   velocity: np.ndarray = None,
                   rotation: np.ndarray = None,
                   mass: float = 1.0,
                   material_name: str = "default",
                   collision_shape: CollisionShape = CollisionShape.SPHERE,
                   collision_radius: float = 0.1,
                   collision_size: np.ndarray = None) -> PhysicsObject:
        """Add a physics object to the simulation"""
        
        if velocity is None:
            velocity = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)
        
        material = self.materials.get(material_name, self.materials["default"])
        
        obj = PhysicsObject(
            id=obj_id,
            object_type=object_type,
            position=position.copy(),
            velocity=velocity.copy(),
            rotation=rotation.copy(),
            angular_velocity=np.zeros(3),
            mass=mass,
            material=material,
            collision_shape=collision_shape,
            collision_radius=collision_radius,
            collision_size=collision_size
        )
        
        self.objects[obj_id] = obj
        logger.debug(f"Added physics object: {obj_id}")
        
        return obj
    
    def add_constraint(self, 
                      constraint_id: str,
                      obj1_id: str,
                      obj2_id: str,
                      constraint_type: str,
                      **parameters) -> Constraint:
        """Add a constraint between two objects"""
        
        constraint = Constraint(
            id=constraint_id,
            object1_id=obj1_id,
            object2_id=obj2_id,
            constraint_type=constraint_type,
            parameters=parameters
        )
        
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint_id}")
        
        return constraint
    
    def apply_force(self, obj_id: str, force: np.ndarray, point: np.ndarray = None):
        """Apply force to an object at a specific point"""
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        obj.forces += force
        
        # Apply torque if force is applied at a point
        if point is not None:
            r = point - obj.position
            torque = np.cross(r, force)
            obj.torques += torque
    
    def apply_impulse(self, obj_id: str, impulse: np.ndarray, point: np.ndarray = None):
        """Apply impulse to an object"""
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        obj.velocity += impulse / obj.mass
        
        # Apply angular impulse if impulse is applied at a point
        if point is not None:
            r = point - obj.position
            angular_impulse = np.cross(r, impulse)
            # Simplified moment of inertia for sphere
            moment_of_inertia = 0.4 * obj.mass * obj.collision_radius**2
            obj.angular_velocity += angular_impulse / moment_of_inertia
    
    def step(self):
        """Simulate one physics step"""
        self.frame_count += 1
        
        # Multiple substeps for stability
        dt = self.time_step / self.substeps
        
        for _ in range(self.substeps):
            self._substep(dt)
    
    def _substep(self, dt: float):
        """Single physics substep"""
        # Update broad phase collision detection
        self._update_broad_phase()
        
        # Apply forces and integrate
        self._apply_forces(dt)
        
        # Solve constraints
        self._solve_constraints()
        
        # Resolve collisions
        self._resolve_collisions()
        
        # Update positions and velocities
        self._integrate(dt)
    
    def _update_broad_phase(self):
        """Update broad phase collision detection grid"""
        self.broad_phase_grid.clear()
        
        for obj_id, obj in self.objects.items():
            if not obj.active:
                continue
            
            # Calculate grid cell for object
            grid_pos = (obj.position / self.grid_size).astype(int)
            grid_key = tuple(grid_pos)
            
            if grid_key not in self.broad_phase_grid:
                self.broad_phase_grid[grid_key] = []
            
            self.broad_phase_grid[grid_key].append(obj_id)
    
    def _apply_forces(self, dt: float):
        """Apply forces to all objects"""
        for obj in self.objects.values():
            if not obj.active:
                continue
            
            # Apply gravity
            obj.forces += self.gravity * obj.mass
            
            # Apply air resistance
            velocity_magnitude = np.linalg.norm(obj.velocity)
            if velocity_magnitude > 0:
                air_resistance_force = -self.air_resistance * velocity_magnitude * obj.velocity
                obj.forces += air_resistance_force
            
            # Apply damping
            obj.forces -= obj.material.damping * obj.velocity
            obj.torques -= obj.material.damping * obj.angular_velocity
    
    def _solve_constraints(self):
        """Solve all constraints"""
        for constraint in self.constraints:
            if constraint.constraint_type == "distance":
                self._solve_distance_constraint(constraint)
            elif constraint.constraint_type == "spring":
                self._solve_spring_constraint(constraint)
            elif constraint.constraint_type == "hinge":
                self._solve_hinge_constraint(constraint)
    
    def _solve_distance_constraint(self, constraint: Constraint):
        """Solve distance constraint between two objects"""
        obj1 = self.objects.get(constraint.object1_id)
        obj2 = self.objects.get(constraint.object2_id)
        
        if not obj1 or not obj2:
            return
        
        target_distance = constraint.parameters.get("distance", 1.0)
        stiffness = constraint.parameters.get("stiffness", 1.0)
        
        # Calculate current distance
        delta = obj2.position - obj1.position
        current_distance = np.linalg.norm(delta)
        
        if current_distance > 0:
            # Calculate correction
            correction = (current_distance - target_distance) / current_distance
            correction_vector = delta * correction * stiffness * 0.5
            
            # Apply correction
            obj1.position += correction_vector
            obj2.position -= correction_vector
    
    def _solve_spring_constraint(self, constraint: Constraint):
        """Solve spring constraint between two objects"""
        obj1 = self.objects.get(constraint.object1_id)
        obj2 = self.objects.get(constraint.object2_id)
        
        if not obj1 or not obj2:
            return
        
        rest_length = constraint.parameters.get("rest_length", 1.0)
        spring_constant = constraint.parameters.get("spring_constant", 100.0)
        damping = constraint.parameters.get("damping", 10.0)
        
        # Calculate spring force
        delta = obj2.position - obj1.position
        distance = np.linalg.norm(delta)
        
        if distance > 0:
            # Spring force
            spring_force = (distance - rest_length) * spring_constant
            force_direction = delta / distance
            
            # Damping force
            relative_velocity = obj2.velocity - obj1.velocity
            damping_force = np.dot(relative_velocity, force_direction) * damping
            
            # Total force
            total_force = (spring_force + damping_force) * force_direction
            
            # Apply forces
            obj1.forces += total_force
            obj2.forces -= total_force
    
    def _solve_hinge_constraint(self, constraint: Constraint):
        """Solve hinge constraint between two objects"""
        obj1 = self.objects.get(constraint.object1_id)
        obj2 = self.objects.get(constraint.object2_id)
        
        if not obj1 or not obj2:
            return
        
        axis = constraint.parameters.get("axis", np.array([0, 1, 0]))
        max_angle = constraint.parameters.get("max_angle", np.pi)
        
        # Calculate relative rotation
        relative_rotation = obj2.rotation @ obj1.rotation.T
        
        # Extract rotation around axis
        # This is a simplified implementation
        angle = np.arccos(np.clip(np.trace(relative_rotation) - 1, -1, 1)) / 2
        
        if abs(angle) > max_angle:
            # Apply angular correction
            correction_angle = (angle - max_angle) if angle > 0 else (angle + max_angle)
            correction_axis = np.cross(axis, np.array([1, 0, 0]))
            if np.linalg.norm(correction_axis) < 0.1:
                correction_axis = np.cross(axis, np.array([0, 1, 0]))
            correction_axis = correction_axis / np.linalg.norm(correction_axis)
            
            # Apply correction
            obj2.angular_velocity += correction_axis * correction_angle * 10.0
    
    def _resolve_collisions(self):
        """Resolve collisions between objects"""
        self.collision_checks = 0
        
        # Check collisions in each grid cell
        for grid_cell, object_ids in self.broad_phase_grid.items():
            if len(object_ids) < 2:
                continue
            
            # Check all pairs in this cell
            for i in range(len(object_ids)):
                for j in range(i + 1, len(object_ids)):
                    obj1_id = object_ids[i]
                    obj2_id = object_ids[j]
                    
                    self.collision_checks += 1
                    self._check_and_resolve_collision(obj1_id, obj2_id)
    
    def _check_and_resolve_collision(self, obj1_id: str, obj2_id: str):
        """Check and resolve collision between two objects"""
        obj1 = self.objects[obj1_id]
        obj2 = self.objects[obj2_id]
        
        if not obj1.active or not obj2.active:
            return
        
        # Calculate collision based on shape
        collision_info = self._check_collision(obj1, obj2)
        
        if collision_info:
            self._resolve_collision(obj1, obj2, collision_info)
    
    def _check_collision(self, obj1: PhysicsObject, obj2: PhysicsObject) -> Optional[Dict]:
        """Check collision between two objects"""
        # Simplified collision detection for spheres
        if (obj1.collision_shape == CollisionShape.SPHERE and 
            obj2.collision_shape == CollisionShape.SPHERE):
            
            delta = obj2.position - obj1.position
            distance = np.linalg.norm(delta)
            min_distance = obj1.collision_radius + obj2.collision_radius
            
            if distance < min_distance:
                return {
                    'normal': delta / distance if distance > 0 else np.array([0, 1, 0]),
                    'penetration': min_distance - distance,
                    'point': obj1.position + (delta / distance) * obj1.collision_radius
                }
        
        return None
    
    def _resolve_collision(self, obj1: PhysicsObject, obj2: PhysicsObject, collision_info: Dict):
        """Resolve collision between two objects"""
        normal = collision_info['normal']
        penetration = collision_info['penetration']
        
        # Separate objects
        separation = normal * penetration * 0.5
        obj1.position -= separation
        obj2.position += separation
        
        # Calculate relative velocity
        relative_velocity = obj2.velocity - obj1.velocity
        
        # Calculate impulse
        restitution = min(obj1.material.restitution, obj2.material.restitution)
        friction = min(obj1.material.friction, obj2.material.friction)
        
        # Normal impulse
        normal_velocity = np.dot(relative_velocity, normal)
        if normal_velocity > 0:
            return  # Objects are moving apart
        
        # Calculate impulse magnitude
        impulse_magnitude = -(1 + restitution) * normal_velocity
        impulse_magnitude /= 1/obj1.mass + 1/obj2.mass
        
        # Apply impulse
        impulse = normal * impulse_magnitude
        obj1.velocity -= impulse / obj1.mass
        obj2.velocity += impulse / obj2.mass
        
        # Friction impulse (simplified)
        tangent_velocity = relative_velocity - normal * normal_velocity
        if np.linalg.norm(tangent_velocity) > 0.01:
            tangent_direction = tangent_velocity / np.linalg.norm(tangent_velocity)
            friction_impulse = tangent_direction * impulse_magnitude * friction
            
            obj1.velocity -= friction_impulse / obj1.mass
            obj2.velocity += friction_impulse / obj2.mass
    
    def _integrate(self, dt: float):
        """Integrate positions and velocities"""
        for obj in self.objects.values():
            if not obj.active:
                continue
            
            # Update velocity
            acceleration = obj.forces / obj.mass
            obj.velocity += acceleration * dt
            
            # Update angular velocity
            angular_acceleration = obj.torques / (0.4 * obj.mass * obj.collision_radius**2)
            obj.angular_velocity += angular_acceleration * dt
            
            # Limit maximum velocity
            velocity_magnitude = np.linalg.norm(obj.velocity)
            if velocity_magnitude > self.max_velocity:
                obj.velocity = obj.velocity / velocity_magnitude * self.max_velocity
            
            # Update position
            obj.position += obj.velocity * dt
            
            # Update rotation (simplified)
            angular_speed = np.linalg.norm(obj.angular_velocity)
            if angular_speed > 0:
                axis = obj.angular_velocity / angular_speed
                angle = angular_speed * dt
                rotation_matrix = self._rotation_matrix_from_axis_angle(axis, angle)
                obj.rotation = rotation_matrix @ obj.rotation
            
            # Reset forces and torques
            obj.forces = np.zeros(3)
            obj.torques = np.zeros(3)
    
    def _rotation_matrix_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix from axis and angle"""
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = axis * np.sin(angle / 2)
        
        return np.array([
            [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
        ])
    
    def get_object_state(self, obj_id: str) -> Optional[Dict]:
        """Get current state of an object"""
        if obj_id not in self.objects:
            return None
        
        obj = self.objects[obj_id]
        return {
            'position': obj.position.copy(),
            'velocity': obj.velocity.copy(),
            'rotation': obj.rotation.copy(),
            'angular_velocity': obj.angular_velocity.copy(),
            'active': obj.active
        }
    
    def set_object_state(self, obj_id: str, position: np.ndarray = None, 
                        velocity: np.ndarray = None, rotation: np.ndarray = None):
        """Set state of an object"""
        if obj_id not in self.objects:
            return
        
        obj = self.objects[obj_id]
        
        if position is not None:
            obj.position = position.copy()
        if velocity is not None:
            obj.velocity = velocity.copy()
        if rotation is not None:
            obj.rotation = rotation.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'frame_count': self.frame_count,
            'object_count': len(self.objects),
            'constraint_count': len(self.constraints),
            'collision_checks': self.collision_checks,
            'active_objects': sum(1 for obj in self.objects.values() if obj.active)
        }
    
    def save_state(self, filename: str):
        """Save physics state to file"""
        state = {
            'objects': {},
            'constraints': [],
            'frame_count': self.frame_count
        }
        
        for obj_id, obj in self.objects.items():
            state['objects'][obj_id] = {
                'position': obj.position.tolist(),
                'velocity': obj.velocity.tolist(),
                'rotation': obj.rotation.tolist(),
                'angular_velocity': obj.angular_velocity.tolist(),
                'mass': obj.mass,
                'material_name': obj.material.name,
                'active': obj.active
            }
        
        for constraint in self.constraints:
            state['constraints'].append({
                'id': constraint.id,
                'object1_id': constraint.object1_id,
                'object2_id': constraint.object2_id,
                'constraint_type': constraint.constraint_type,
                'parameters': constraint.parameters
            })
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Physics state saved to {filename}")
    
    def load_state(self, filename: str):
        """Load physics state from file"""
        with open(filename, 'r') as f:
            state = json.load(f)
        
        # Clear current state
        self.objects.clear()
        self.constraints.clear()
        
        # Load objects
        for obj_id, obj_data in state['objects'].items():
            material = self.materials.get(obj_data['material_name'], self.materials['default'])
            
            obj = PhysicsObject(
                id=obj_id,
                object_type=PhysicsObjectType.RIGID_BODY,  # Default type
                position=np.array(obj_data['position']),
                velocity=np.array(obj_data['velocity']),
                rotation=np.array(obj_data['rotation']),
                angular_velocity=np.array(obj_data['angular_velocity']),
                mass=obj_data['mass'],
                material=material,
                collision_shape=CollisionShape.SPHERE,  # Default shape
                active=obj_data['active']
            )
            
            self.objects[obj_id] = obj
        
        # Load constraints
        for constraint_data in state['constraints']:
            constraint = Constraint(
                id=constraint_data['id'],
                object1_id=constraint_data['object1_id'],
                object2_id=constraint_data['object2_id'],
                constraint_type=constraint_data['constraint_type'],
                parameters=constraint_data['parameters']
            )
            
            self.constraints.append(constraint)
        
        self.frame_count = state.get('frame_count', 0)
        
        logger.info(f"Physics state loaded from {filename}")

def create_sample_scene(engine: AdvancedPhysicsEngine):
    """Create a sample physics scene for testing"""
    
    # Create ground plane
    ground = engine.add_object(
        "ground",
        PhysicsObjectType.RIGID_BODY,
        position=np.array([0, -2, 0]),
        mass=0,  # Static object
        material_name="metal",
        collision_shape=CollisionShape.BOX,
        collision_size=np.array([10, 0.5, 10])
    )
    
    # Create some bouncing balls
    for i in range(5):
        ball = engine.add_object(
            f"ball_{i}",
            PhysicsObjectType.RIGID_BODY,
            position=np.array([i * 2 - 4, 5, 0]),
            velocity=np.array([0, 0, 0]),
            mass=1.0,
            material_name="rubber",
            collision_shape=CollisionShape.SPHERE,
            collision_radius=0.5
        )
    
    # Create a pendulum
    pendulum_base = engine.add_object(
        "pendulum_base",
        PhysicsObjectType.RIGID_BODY,
        position=np.array([6, 3, 0]),
        mass=0,  # Static
        material_name="metal"
    )
    
    pendulum_bob = engine.add_object(
        "pendulum_bob",
        PhysicsObjectType.RIGID_BODY,
        position=np.array([6, 1, 0]),
        mass=2.0,
        material_name="metal",
        collision_shape=CollisionShape.SPHERE,
        collision_radius=0.3
    )
    
    # Add pendulum constraint
    engine.add_constraint(
        "pendulum_constraint",
        "pendulum_base",
        "pendulum_bob",
        "distance",
        distance=2.0,
        stiffness=1.0
    )
    
    logger.info("Sample physics scene created")

if __name__ == "__main__":
    # Test the physics engine
    engine = AdvancedPhysicsEngine()
    
    # Create sample scene
    create_sample_scene(engine)
    
    # Run simulation
    print("Running physics simulation...")
    for i in range(300):  # 5 seconds at 60 FPS
        engine.step()
        
        if i % 60 == 0:  # Print every second
            stats = engine.get_performance_stats()
            print(f"Frame {i}: {stats['active_objects']} active objects, "
                  f"{stats['collision_checks']} collision checks")
    
    print("Simulation complete!")