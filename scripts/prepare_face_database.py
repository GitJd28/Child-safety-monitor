# scripts/prepare_face_database.py
"""
Select specific LFW identities to simulate a family:
- Parent 1 (e.g., George_W_Bush - 530 images, very recognizable)
- Parent 2 (e.g., Colin_Powell - 236 images)
- Nanny    (e.g., Tony_Blair - 144 images)
- Unknown  (everyone else = strangers)
"""

import os
import shutil
import random
from pathlib import Path

def get_lfw_identities_with_counts(lfw_path):
    """Get all LFW identities sorted by number of images"""
    identities = {}
    
    for person_dir in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person_dir)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
            identities[person_dir] = len(images)
    
    # Sort by count (most images first - better for training)
    sorted_identities = sorted(identities.items(), key=lambda x: x[1], reverse=True)
    return sorted_identities


def prepare_known_faces(lfw_path, output_dir, train_split=0.8):
    """
    Create known faces database from LFW
    
    Structure created:
    processed/known_faces/
        parent1/          ← Images of George_W_Bush
            train/        ← 80% for enrollment
            test/         ← 20% for verification testing
        parent2/          ← Images of Colin_Powell
        nanny/            ← Images of Tony_Blair
    
    processed/unknown_faces/
        person1/          ← 20 random LFW identities
        person2/
        ...
    """
    
    print("Preparing face database...")
    print("="*50)
    
    # Get identities with enough images
    identities = get_lfw_identities_with_counts(lfw_path)
    
    print(f"\nTop 10 LFW identities by image count:")
    for name, count in identities[:10]:
        print(f"  {name}: {count} images")
    
    # Select "known" family members
    # Using well-represented identities for better testing
    known_family = {
        'parent1': identities[0][0],   # Most images (George_W_Bush: 530)
        'parent2': identities[1][0],   # Second most (Colin_Powell: 236)
        'nanny':   identities[2][0],   # Third most (Tony_Blair: 144)
    }
    
    print(f"\n📋 Simulated Family:")
    for role, identity in known_family.items():
        count = dict(identities)[identity]
        print(f"  {role}: {identity} ({count} images)")
    
    # Create known faces directories
    for role, identity in known_family.items():
        src_path = os.path.join(lfw_path, identity)
        images = sorted([f for f in os.listdir(src_path) if f.endswith('.jpg')])
        
        # Split: train for enrollment, test for verification
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create train dir (enrollment database)
        train_dir = os.path.join(output_dir, 'known_faces', role, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        # Create test dir (for accuracy testing)
        test_dir = os.path.join(output_dir, 'known_faces', role, 'test')
        os.makedirs(test_dir, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(train_dir, img)
            )
        
        for img in test_images:
            shutil.copy(
                os.path.join(src_path, img),
                os.path.join(test_dir, img)
            )
        
        print(f"\n {role} ({identity}):")
        print(f"   Enrollment images: {len(train_images)}")
        print(f"   Test images: {len(test_images)}")
    
    # Create unknown faces pool (20 random people NOT in family)
    print(f"\n Creating unknown faces pool (for testing stranger detection)...")
    
    known_names = set(known_family.values())
    unknown_candidates = [
        (name, count) for name, count in identities
        if name not in known_names and count >= 5  # At least 5 images
    ]
    
    # Pick 20 random strangers
    random.seed(42)  # Reproducible
    selected_strangers = random.sample(unknown_candidates, min(20, len(unknown_candidates)))
    
    for stranger_name, count in selected_strangers:
        src_path = os.path.join(lfw_path, stranger_name)
        dest_path = os.path.join(output_dir, 'unknown_faces', stranger_name)
        
        if os.path.exists(dest_path):
            continue
        
        shutil.copytree(src_path, dest_path)
    
    print(f" Unknown faces pool: {len(selected_strangers)} strangers")
    print(f"   Used for testing: 'stranger detected' alert accuracy")
    
    return known_family


def prepare_child_faces(utk_path, output_dir):
    """
    Filter UTKFace for children (age 4-12)
    and create a 'child1' identity in known faces
    """
    
    if not os.path.exists(utk_path):
        print("\n  UTKFace not found. Skipping child faces.")
        print("   Download UTKFace manually from:")
        print("   https://susanqq.github.io/UTKFace/")
        return
    
    print("\n Filtering child faces from UTKFace (age 4-12)...")
    
    all_images = [f for f in os.listdir(utk_path) if f.endswith('.jpg')]
    child_images = []
    
    for img_name in all_images:
        try:
            age = int(img_name.split('_')[0])
            if 4 <= age <= 12:
                child_images.append((img_name, age))
        except (ValueError, IndexError):
            continue
    
    print(f" Found {len(child_images)} child images (age 4-12)")
    
    if len(child_images) == 0:
        print("   No child images found. Check UTKFace download.")
        return
    
    # Group by approximate age to simulate 2 different children
    younger = [(img, age) for img, age in child_images if age <= 7]
    older = [(img, age) for img, age in child_images if 8 <= age <= 12]
    
    # Create child1 (younger, age 4-7)
    child1_dir = os.path.join(output_dir, 'known_faces', 'child1', 'train')
    os.makedirs(child1_dir, exist_ok=True)
    
    # Use first 30 images of younger children
    for img_name, age in younger[:30]:
        shutil.copy(
            os.path.join(utk_path, img_name),
            os.path.join(child1_dir, img_name)
        )
    
    print(f" Child1 (age 4-7): {min(30, len(younger))} images enrolled")
    
    # Create child2 (older, age 8-12)
    child2_dir = os.path.join(output_dir, 'known_faces', 'child2', 'train')
    os.makedirs(child2_dir, exist_ok=True)
    
    for img_name, age in older[:30]:
        shutil.copy(
            os.path.join(utk_path, img_name),
            os.path.join(child2_dir, img_name)
        )
    
    print(f" Child2 (age 8-12): {min(30, len(older))} images enrolled")


def print_summary(output_dir):
    """Print final database summary"""
    print("\n" + "="*50)
    print("FACE DATABASE SUMMARY")
    print("="*50)
    
    known_dir = os.path.join(output_dir, 'known_faces')
    unknown_dir = os.path.join(output_dir, 'unknown_faces')
    
    if os.path.exists(known_dir):
        print("\n Known Faces (Family + Child):")
        for person in os.listdir(known_dir):
            person_path = os.path.join(known_dir, person)
            
            train_path = os.path.join(person_path, 'train')
            test_path = os.path.join(person_path, 'test')
            
            train_count = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
            test_count = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
            
            print(f"   {person:12}: {train_count} train, {test_count} test")
    
    if os.path.exists(unknown_dir):
        strangers = os.listdir(unknown_dir)
        total_stranger_imgs = sum(
            len(os.listdir(os.path.join(unknown_dir, s)))
            for s in strangers
        )
        print(f"\n Unknown Faces (Strangers):")
        print(f"   {len(strangers)} different strangers")
        print(f"   {total_stranger_imgs} total stranger images")
    
    print("\n Face database ready for recognition!")


if __name__ == "__main__":
    
    LFW_PATH = "data/raw/faces/lfw/lfw"
    UTK_PATH = "data/raw/faces/utkface"
    OUTPUT_DIR = "data/processed"
    
    if not os.path.exists(LFW_PATH):
        print(" LFW not found. Run download_datasets.py first.")
        exit(1)
    
    # Prepare known and unknown faces from LFW
    family = prepare_known_faces(LFW_PATH, OUTPUT_DIR)
    
    # Prepare child faces from UTKFace (if available)
    prepare_child_faces(UTK_PATH, OUTPUT_DIR)
    
    # Summary
    print_summary(OUTPUT_DIR)