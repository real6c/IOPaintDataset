import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from iopaint.helper import pil_to_bytes
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
    except:
        pass
    return 0


def glob_images_recursive(path: Path) -> Dict[str, Path]:
    """Recursively find all image files in a directory and its subdirectories."""
    res = {}
    if path.is_file():
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            res[path.stem] = path
    elif path.is_dir():
        for it in path.rglob("*"):
            if it.is_file() and it.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                res[it.stem] = it
    return res


def glob_images(path: Path) -> Dict[str, Path]:
    # png/jpg/jpeg
    if path.is_file():
        return {path.stem: path}
    elif path.is_dir():
        res = {}
        for it in path.glob("*.*"):
            if it.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                res[it.stem] = it
        return res


def find_mask_for_image(image_path: Path, mask_dir: Path, mask_suffix: str = "_mask") -> Optional[Path]:
    """Find the corresponding mask file for an image using the mask suffix."""
    # Get the relative path from the image to its root directory
    image_stem = image_path.stem
    
    # Try to find mask with suffix
    mask_filename = f"{image_stem}{mask_suffix}.png"
    mask_path = mask_dir / mask_filename
    
    if mask_path.exists():
        return mask_path
    
    # Fallback: try without suffix
    mask_filename_no_suffix = f"{image_stem}.png"
    mask_path_no_suffix = mask_dir / mask_filename_no_suffix
    
    if mask_path_no_suffix.exists():
        return mask_path_no_suffix
    
    return None


def get_relative_path(file_path: Path, base_dir: Path) -> Path:
    """Get the relative path of a file from a base directory."""
    try:
        return file_path.relative_to(base_dir)
    except ValueError:
        return file_path


class ThreadLocalModelManager:
    """Thread-local storage for model managers to ensure thread safety."""
    def __init__(self, model_name: str, device):
        self.model_name = model_name
        self.device = device
        self.thread_local = threading.local()
    
    def get_model(self):
        """Get or create a model manager for the current thread."""
        if not hasattr(self.thread_local, 'model_manager'):
            logger.info(f"Loading model for thread {threading.current_thread().name}")
            self.thread_local.model_manager = ModelManager(name=self.model_name, device=self.device)
            
            # Warm up the model with a dummy inference
            logger.info(f"Warming up model for thread {threading.current_thread().name}")
            try:
                # Create a small dummy image for warm-up
                dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                dummy_mask = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
                dummy_mask[dummy_mask >= 127] = 255
                dummy_mask[dummy_mask < 127] = 0
                
                # Simple inpaint request for warm-up
                from iopaint.schema import InpaintRequest
                dummy_request = InpaintRequest()
                
                # Run warm-up inference
                warmup_start = time.time()
                _ = self.thread_local.model_manager(dummy_img, dummy_mask, dummy_request)
                warmup_time = time.time() - warmup_start
                logger.info(f"Model warm-up completed for thread {threading.current_thread().name} in {warmup_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Model warm-up failed for thread {threading.current_thread().name}: {str(e)}")
        
        return self.thread_local.model_manager


def process_single_image(args):
    """Process a single image with thread-local model instance."""
    (stem, image_p, mask_p, model_manager_factory, inpaint_request, 
     output, image_base_dir, recursive, concat, verbose) = args
    
    start_time = time.time()
    thread_name = threading.current_thread().name
    
    try:
        # Get thread-local model manager
        model_start = time.time()
        model_manager = model_manager_factory.get_model()
        model_load_time = time.time() - model_start
        
        # Load and process image
        load_start = time.time()
        infos = Image.open(image_p).info
        img = np.array(Image.open(image_p).convert("RGB"))
        mask_img = np.array(Image.open(mask_p).convert("L"))
        load_time = time.time() - load_start

        if mask_img.shape[:2] != img.shape[:2]:
            logger.debug(
                f"resize mask {mask_p.name} to image {image_p.name} size: {img.shape[:2]}"
            )
            mask_img = cv2.resize(
                mask_img,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_img[mask_img >= 127] = 255
        mask_img[mask_img < 127] = 0

        # Process with model
        inference_start = time.time()
        inpaint_result = model_manager(img, mask_img, inpaint_request)
        inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
        inference_time = time.time() - inference_start
        
        if concat:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
            inpaint_result = cv2.hconcat([img, mask_img, inpaint_result])

        # Save result
        save_start = time.time()
        img_bytes = pil_to_bytes(Image.fromarray(inpaint_result), "png", 100, infos)
        
        # Preserve directory structure in output
        if recursive and image_base_dir.is_dir():
            rel_path = get_relative_path(image_p, image_base_dir)
            output_path = output / rel_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            save_p = output_path / f"{stem}.png"
        else:
            save_p = output / f"{stem}.png"
        
        with open(save_p, "wb") as fw:
            fw.write(img_bytes)
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        
        # Log performance metrics with more detail
        if verbose:
            logger.debug(f"Thread {thread_name} - {stem}: ModelLoad={model_load_time:.2f}s, Load={load_time:.2f}s, Inference={inference_time:.2f}s, Save={save_time:.2f}s, Total={total_time:.2f}s")

        return True, stem
    except Exception as e:
        logger.error(f"Error processing {stem} in thread {thread_name}: {str(e)}")
        return False, stem


def batch_inpaint(
    model: str,
    device,
    image: Path,
    mask: Path,
    output: Path,
    config: Optional[Path] = None,
    concat: bool = False,
    recursive: bool = False,
    mask_suffix: str = "_mask",
    num_workers: int = 4,
    verbose: bool = False,
):
    if image.is_dir() and output.is_file():
        logger.error(
            "invalid --output: when image is a directory, output should be a directory"
        )
        exit(-1)
    output.mkdir(parents=True, exist_ok=True)

    # Use recursive search if requested
    if recursive:
        image_paths = glob_images_recursive(image)
    else:
        image_paths = glob_images(image)
    
    if len(image_paths) == 0:
        logger.error("invalid --image: empty image folder")
        exit(-1)

    # Handle mask directory structure
    if mask.is_dir():
        # Check if masks are in a subdirectory structure
        mask_paths = {}
        for stem, image_p in image_paths.items():
            # Get relative path from image to its root directory
            rel_path = get_relative_path(image_p, image)
            
            # Try to find mask in the mask directory structure
            # First, try the same relative path in mask directory
            potential_mask_dir = mask / rel_path.parent
            mask_file = find_mask_for_image(image_p, potential_mask_dir, mask_suffix)
            
            if mask_file is None:
                # Try looking in a 'masks' subdirectory
                masks_subdir = potential_mask_dir / "masks"
                if masks_subdir.exists():
                    mask_file = find_mask_for_image(image_p, masks_subdir, mask_suffix)
            
            if mask_file is not None:
                mask_paths[stem] = mask_file
            else:
                logger.warning(f"No mask found for {image_p}")
    else:
        # Single mask file for all images
        mask_paths = {stem: mask for stem in image_paths.keys()}

    if len(mask_paths) == 0:
        logger.error("invalid --mask: no valid masks found")
        exit(-1)

    if config is None:
        inpaint_request = InpaintRequest()
        logger.info(f"Using default config: {inpaint_request}")
    else:
        with open(config, "r", encoding="utf-8") as f:
            inpaint_request = InpaintRequest(**json.load(f))
        logger.info(f"Using config: {inpaint_request}")

    console = Console()
    
    # Create thread-local model manager factory
    model_manager_factory = ThreadLocalModelManager(model, device)
    
    logger.info(f"Starting parallel processing with {num_workers} workers")
    logger.info(f"Initial GPU memory usage: {get_gpu_memory_usage():.2f} GB")

    # Pre-warmup phase: Load and warm up all model instances
    logger.info("Pre-warming up all model instances...")
    warmup_start = time.time()
    
    def warmup_worker(worker_id):
        """Warm up a single worker's model instance."""
        thread_name = f"WarmupWorker-{worker_id}"
        threading.current_thread().name = thread_name
        try:
            model_manager = model_manager_factory.get_model()
            logger.info(f"Worker {worker_id} model loaded and warmed up")
            return True
        except Exception as e:
            logger.error(f"Failed to warm up worker {worker_id}: {str(e)}")
            return False
    
    # Warm up all workers in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        warmup_futures = [executor.submit(warmup_worker, i) for i in range(num_workers)]
        warmup_results = [future.result() for future in warmup_futures]
    
    warmup_time = time.time() - warmup_start
    successful_warmups = sum(warmup_results)
    logger.info(f"Pre-warmup completed: {successful_warmups}/{num_workers} workers ready in {warmup_time:.2f}s")
    logger.info(f"GPU memory after warmup: {get_gpu_memory_usage():.2f} GB")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Batch processing...", total=len(image_paths))
        
        # Prepare arguments for parallel processing
        processing_args = []
        for stem, image_p in image_paths.items():
            if stem not in mask_paths:
                progress.log(f"mask for {image_p} not found")
                progress.update(task, advance=1)
                continue
            mask_p = mask_paths.get(stem)
            
            processing_args.append((
                stem, image_p, mask_p, model_manager_factory, inpaint_request,
                output, image, recursive, concat, verbose
            ))
        
        # Process images in parallel
        completed_count = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_stem = {}
            for args in processing_args:
                future = executor.submit(process_single_image, args)
                future_to_stem[future] = args[0]  # stem
            
            # Process completed tasks
            for future in as_completed(future_to_stem):
                stem = future_to_stem[future]
                try:
                    success, processed_stem = future.result()
                    if success:
                        completed_count += 1
                    progress.update(task, advance=1)
                    
                    # Periodic memory cleanup and logging
                    if completed_count % 10 == 0:
                        torch_gc()
                        gpu_memory = get_gpu_memory_usage()
                        elapsed_time = time.time() - start_time
                        rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"Progress: {completed_count}/{len(image_paths)} images, "
                                  f"GPU Memory: {gpu_memory:.2f} GB, "
                                  f"Rate: {rate:.2f} images/sec")
                        
                except Exception as e:
                    logger.error(f"Error processing {stem}: {str(e)}")
                    progress.update(task, advance=1)
        
        total_time = time.time() - start_time
        final_gpu_memory = get_gpu_memory_usage()
        avg_rate = len(image_paths) / total_time if total_time > 0 else 0
        
        logger.info(f"Completed processing {completed_count}/{len(image_paths)} images")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average rate: {avg_rate:.2f} images/sec")
        logger.info(f"Final GPU memory usage: {final_gpu_memory:.2f} GB")
