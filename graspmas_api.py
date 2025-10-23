"""
GraspMAS API - Simple interface for zero-shot grasp detection

Usage:
    from graspmas_api import detect_grasp_from_image
    
    result = detect_grasp_from_image(
        image_path="rubber_ducky.png",
        prompt="Grasp the rubber duck"
    )
    
    print(f"Grasp center: ({result['center_x']:.1f}, {result['center_y']:.1f})")
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import asyncio
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Add GraspMAS to path
SCRIPT_DIR = Path(__file__).parent.absolute()
GRASPMAS_DIR = SCRIPT_DIR / "GraspMAS"
if str(GRASPMAS_DIR) not in sys.path:
    sys.path.insert(0, str(GRASPMAS_DIR))

from agents.graspmas import GraspMAS


def detect_grasp_from_image(
    image_path: str,
    prompt: str,
    api_key_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_refinement_rounds: int = 3
) -> Dict:
    """
    Detect grasp pose from an image using natural language prompt.
    
    Parameters
    ----------
    image_path : str
        Path to input image (.png, .jpg, .jpeg, .npz supported)
    prompt : str
        Natural language grasp query (e.g., "Grasp the rubber duck")
    api_key_file : str, optional
        Path to OpenAI API key file (default: GraspMAS/api.key)
    output_dir : str, optional
        Directory to save output images (default: graspmas_output/)
    max_refinement_rounds : int, optional
        Maximum refinement iterations (default: 3)
    
    Returns
    -------
    dict
        Grasp detection results with keys:
        - success: bool - Whether detection succeeded
        - center_x: float - X coordinate of grasp center (pixels, from left)
        - center_y: float - Y coordinate of grasp center (pixels, from top)
        - width: float - Grasp width (pixels)
        - height: float - Grasp height (pixels)
        - angle: float - Grasp angle (degrees, 0° = horizontal)
        - quality: float - Confidence score (0.0 to 1.0)
        - image_width: int - Original image width
        - image_height: int - Original image height
        - output_files: dict - Paths to saved visualization images
        - prompt: str - The query used
    """
    # Setup paths
    image_path = Path(image_path)
    if not image_path.exists():
        return {
            "success": False,
            "error": f"Image file not found: {image_path}",
            "prompt": prompt
        }
    
    # Default API key location
    if api_key_file is None:
        api_key_file = GRASPMAS_DIR / "api.key"
    
    if not Path(api_key_file).exists():
        return {
            "success": False,
            "error": f"API key file not found: {api_key_file}. Create it with your OpenAI API key.",
            "prompt": prompt
        }
    
    # Setup output directory with timestamp
    if output_dir is None:
        output_dir = SCRIPT_DIR / "graspmas_output"
    output_dir = Path(output_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{image_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process image
    try:
        img, img_path = _load_image(image_path, run_dir)
        img_height, img_width = img.shape[:2]
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load image: {str(e)}",
            "prompt": prompt
        }
    
    # Run GraspMAS detection
    try:
        grasp_pose, vis_path = _run_graspmas(
            str(img_path),
            prompt,
            str(api_key_file),
            max_refinement_rounds
        )
        
        if grasp_pose is None or len(grasp_pose) != 6:
            return {
                "success": False,
                "error": "GraspMAS failed to detect grasp",
                "prompt": prompt
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"GraspMAS execution failed: {str(e)}",
            "prompt": prompt
        }
    
    # Parse grasp pose
    quality, center_x, center_y, width, height, angle = grasp_pose
    
    # Generate output visualizations
    output_files = _create_visualizations(
        img,
        grasp_pose,
        run_dir
    )
    
    # Return structured result
    return {
        "success": True,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": float(width),
        "height": float(height),
        "angle": float(angle),
        "quality": float(quality),
        "image_width": img_width,
        "image_height": img_height,
        "coordinate_system": "top-left origin (OpenCV convention)",
        "output_files": output_files,
        "output_directory": str(run_dir),
        "prompt": prompt
    }


def _load_image(image_path: Path, output_dir: Path) -> Tuple[np.ndarray, Path]:
    """Load image from various formats and save as JPG."""
    # Handle NPZ files (with RGB data)
    if image_path.suffix.lower() == '.npz':
        data = np.load(image_path)
        # Try common keys for RGB data
        for key in ['left_rgb', 'rgb', 'image', 'color']:
            if key in data:
                img = data[key]
                break
        else:
            raise ValueError(f"No RGB data found in NPZ file. Available keys: {list(data.keys())}")
        
        img = Image.fromarray(img)
    else:
        # Load standard image formats
        img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save as JPG in output directory (for GraspMAS input)
    output_path = output_dir / "input.jpg"
    img.save(output_path, quality=95)
    
    return np.array(img), output_path


def _run_graspmas(
    image_path: str,
    prompt: str,
    api_key_file: str,
    max_rounds: int
) -> Tuple[Optional[list], Optional[str]]:
    """Run GraspMAS asynchronously."""
    async def _async_detect():
        graspmas = GraspMAS(api_file=api_key_file, max_round=max_rounds)
        save_path, grasp_pose = await graspmas.query(prompt, image_path)
        return grasp_pose, save_path
    
    # Run async function
    return asyncio.run(_async_detect())


def _create_visualizations(
    img: np.ndarray,
    grasp_pose: list,
    output_dir: Path
) -> Dict[str, str]:
    """Create and save visualization images."""
    quality, cx, cy, w, h, angle = grasp_pose
    
    output_files = {}
    
    # 1. Original image
    original_path = output_dir / "original.png"
    Image.fromarray(img).save(original_path)
    output_files['original'] = str(original_path)
    
    # 2. Image with grasp rectangle overlay
    img_with_grasp = img.copy()
    
    # Draw oriented rectangle
    center = (int(cx), int(cy))
    size = (int(w), int(h))
    box = cv2.boxPoints((center, size, angle))
    box = np.int32(box)
    
    # Draw rectangle with different colors for gripper jaws
    cv2.line(img_with_grasp, tuple(box[0]), tuple(box[3]), (255, 255, 0), 3)  # Yellow
    cv2.line(img_with_grasp, tuple(box[3]), tuple(box[2]), (255, 0, 255), 3)  # Magenta
    cv2.line(img_with_grasp, tuple(box[2]), tuple(box[1]), (255, 255, 0), 3)  # Yellow
    cv2.line(img_with_grasp, tuple(box[1]), tuple(box[0]), (255, 0, 255), 3)  # Magenta
    
    # Draw center point
    cv2.circle(img_with_grasp, center, 5, (0, 255, 0), -1)
    
    # Add text annotation
    text = f"Q:{quality:.3f} A:{angle:.1f}°"
    cv2.putText(img_with_grasp, text, (int(cx) + 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    grasp_overlay_path = output_dir / "grasp_overlay.png"
    Image.fromarray(img_with_grasp).save(grasp_overlay_path)
    output_files['grasp_overlay'] = str(grasp_overlay_path)
    
    # 3. Localized object region (crop around grasp)
    # Create a bounding box around the grasp rectangle with padding
    padding = int(max(w, h) * 0.5)
    x_min = max(0, int(cx - w/2 - padding))
    x_max = min(img.shape[1], int(cx + w/2 + padding))
    y_min = max(0, int(cy - h/2 - padding))
    y_max = min(img.shape[0], int(cy + h/2 + padding))
    
    localized_img = img[y_min:y_max, x_min:x_max].copy()
    
    # Draw grasp on localized image (adjust coordinates)
    local_center = (int(cx - x_min), int(cy - y_min))
    local_box = box - np.array([x_min, y_min])
    
    cv2.line(localized_img, tuple(local_box[0]), tuple(local_box[3]), (255, 255, 0), 2)
    cv2.line(localized_img, tuple(local_box[3]), tuple(local_box[2]), (255, 0, 255), 2)
    cv2.line(localized_img, tuple(local_box[2]), tuple(local_box[1]), (255, 255, 0), 2)
    cv2.line(localized_img, tuple(local_box[1]), tuple(local_box[0]), (255, 0, 255), 2)
    cv2.circle(localized_img, local_center, 3, (0, 255, 0), -1)
    
    localized_path = output_dir / "localized.png"
    Image.fromarray(localized_img).save(localized_path)
    output_files['localized'] = str(localized_path)
    
    # 4. Input image (the one fed to GraspMAS)
    input_path = output_dir / "input.png"
    if input_path.exists():
        output_files['input'] = str(input_path)
    
    return output_files


def print_result(result: Dict) -> None:
    """Pretty print the grasp detection result."""
    print("\n" + "="*70)
    print("GRASPMAS GRASP DETECTION RESULT")
    print("="*70)
    
    if not result['success']:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        print(f"Prompt: {result['prompt']}")
        return
    
    print(f"✅ SUCCESS")
    print(f"\nPrompt: '{result['prompt']}'")
    print(f"\n📍 Grasp Pose:")
    print(f"   Center: ({result['center_x']:.1f}, {result['center_y']:.1f}) pixels")
    print(f"   Width:  {result['width']:.1f} pixels")
    print(f"   Height: {result['height']:.1f} pixels")
    print(f"   Angle:  {result['angle']:.1f}°")
    print(f"   Quality: {result['quality']:.4f}")
    
    print(f"\n📐 Image Info:")
    print(f"   Size: {result['image_width']} × {result['image_height']} pixels")
    print(f"   Coordinate system: {result['coordinate_system']}")
    
    print(f"\n💾 Output Directory:")
    print(f"   {result['output_directory']}")
    print(f"\n💾 Output Files:")
    for key, path in result['output_files'].items():
        filename = Path(path).name
        print(f"   {key:15s}: {filename}")
    
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the GraspMAS API.
    
    Run this script:
        python graspmas_api.py
    
    Or import and use in your code:
        from graspmas_api import detect_grasp_from_image
        result = detect_grasp_from_image("image.png", "Grasp the object")
    """
    
    # Example 1: PNG image - Simple usage with rubber duck
    print("\n" + "🦆 "*20)
    print("EXAMPLE 1: PNG Image - Rubber Duck")
    print("🦆 "*20 + "\n")
    
    result_png = detect_grasp_from_image(
        image_path="rubber_ducky.png",
        prompt="Grasp the rubber duck"
    )
    
    print_result(result_png)
    
    # Example 2: NPZ file - Camera observation
    print("\n" + "📷 "*20)
    print("EXAMPLE 2: NPZ File - Camera Observation")
    print("📷 "*20 + "\n")
    
    result_npz = detect_grasp_from_image(
        image_path="camera_obs_20251008_162139_624582.npz",
        prompt="Grasp the object in the center of the image"
    )
    
    print_result(result_npz)
    
    # Example 3: Using the result in your code
    print("\n" + "🤖 "*20)
    print("EXAMPLE 3: Using result for robot control")
    print("🤖 "*20 + "\n")
    
    if result_png['success']:
        # Extract grasp parameters
        cx, cy = result_png['center_x'], result_png['center_y']
        angle = result_png['angle']
        width = result_png['width']
        
        print("Sending to robot controller:")
        print(f"  - Move gripper to pixel ({cx:.1f}, {cy:.1f})")
        print(f"  - Rotate to {angle:.1f}°")
        print(f"  - Open gripper to {width:.1f} pixels equivalent")
        print(f"  - Execute grasp with confidence {result_png['quality']:.1%}")
        
        print(f"\n📸 Check visualizations in: {result_png['output_directory']}")
    
    print("\n" + "="*70)
    print("✨ Done! Check the graspmas_output/ directory for results.")
    print("="*70 + "\n")
