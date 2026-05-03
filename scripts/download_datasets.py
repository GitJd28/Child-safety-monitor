# scripts/download_datasets.py
"""
Downloads only what we need:
1. LFW (173MB) - for simulating known/unknown adult faces
2. UTKFace (107MB) - for child face examples
"""

import os
import requests
import urllib.request
from tqdm import tqdm
import zipfile
import tarfile

def download_with_progress(url, filepath):
    """Download file showing progress"""
    
    print(f"\nDownloading: {os.path.basename(filepath)}")
    print(f"From: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"File size: {total_size / (1024*1024):.1f} MB")
        
        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Progress"
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        print(f"✅ Downloaded: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_lfw():
    """Download LFW (Labeled Faces in the Wild)"""
    
    save_dir = "data/raw/faces/lfw"
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, "lfw.tgz")
    extract_dir = os.path.join(save_dir, "lfw")
    
    # Check if already downloaded
    if os.path.exists(extract_dir):
        count = sum(len(files) for _, _, files in os.walk(extract_dir))
        print(f"✅ LFW already exists ({count} files)")
        return extract_dir
    
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    
    success = download_with_progress(url, filepath)
    
    if not success:
        print("\nManual download instructions:")
        print("1. Go to: http://vis-www.cs.umass.edu/lfw/")
        print("2. Click 'All images as gzipped tar file'")
        print(f"3. Save as: {filepath}")
        return None
    
    # Extract
    print("Extracting LFW...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(save_dir)
    
    # Count
    count = sum(len(files) for _, _, files in os.walk(extract_dir))
    print(f"✅ LFW extracted: {count} images")
    print(f"   {len(os.listdir(extract_dir))} identities")
    
    return extract_dir


def download_utkface():
    """
    UTKFace requires manual download (Google Drive link).
    Provide instructions.
    """
    save_dir = "data/raw/faces/utkface"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("UTKFace Dataset - Manual Download Required")
    print("="*60)
    print("\nUTKFace is hosted on Google Drive (requires manual download)")
    print("\nInstructions:")
    print("1. Go to: https://susanqq.github.io/UTKFace/")
    print("2. Click 'Aligned&Cropped Faces' (Part 1, Part 2, Part 3)")
    print("3. Download all 3 parts (total ~107MB)")
    print(f"4. Extract all .jpg files into: {os.path.abspath(save_dir)}")
    print("\nFilename format: [age]_[gender]_[race]_[date].jpg")
    print("Example: 4_0_0_20170109142408075.jpg = 4-year-old male")
    print("\n✅ We only need Part 1 (ages 0-30 approximately)")
    print("   This gives us enough child faces (age 4-12)")
    
    return save_dir


def verify_datasets():
    """Check what's been downloaded"""
    print("\n" + "="*60)
    print("DATASET STATUS")
    print("="*60)
    
    # Check LFW
    lfw_path = "data/raw/faces/lfw/lfw"
    if os.path.exists(lfw_path):
        identities = os.listdir(lfw_path)
        total_images = sum(
            len(files) for _, _, files in os.walk(lfw_path)
        )
        print(f"\n✅ LFW Dataset:")
        print(f"   Identities: {len(identities)}")
        print(f"   Total images: {total_images}")
        print(f"   Location: {lfw_path}")
    else:
        print(f"\n❌ LFW Dataset: Not found at {lfw_path}")
    
    # Check UTKFace
    utk_path = "data/raw/faces/utkface"
    if os.path.exists(utk_path):
        images = [f for f in os.listdir(utk_path) if f.endswith('.jpg')]
        child_images = []
        for img in images:
            try:
                age = int(img.split('_')[0])
                if 4 <= age <= 12:
                    child_images.append(img)
            except:
                pass
        print(f"\n✅ UTKFace Dataset:")
        print(f"   Total images: {len(images)}")
        print(f"   Child images (age 4-12): {len(child_images)}")
        print(f"   Location: {utk_path}")
    else:
        print(f"\n⚠️  UTKFace Dataset: Not found (manual download needed)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Child Safety Monitor - Dataset Downloader")
    print("="*60)
    print("Focused on FACE RECOGNITION only (core feature)")
    print("Audio skipped (future scope)")
    print("="*60)
    
    # Step 1: Download LFW
    print("\n[1/2] LFW Dataset (173MB)...")
    lfw_dir = download_lfw()
    
    # Step 2: Instructions for UTKFace
    print("\n[2/2] UTKFace Dataset...")
    download_utkface()
    
    # Verify
    verify_datasets()