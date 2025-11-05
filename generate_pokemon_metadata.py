"""
Pokemon Metadata Generator

Processes Pokemon images and generates structured metadata JSON files using
BAML's vision API. Each image receives a corresponding metadata file with
visual attributes, categorical information, and text prompts.
"""

import base64
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from baml_client import b
from baml_client.types import PokemonMetadata
from baml_py import Image
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _convert_to_serializable(value: Any) -> Any:
    """
    Recursively convert Pydantic models and enums to JSON-serializable types.

    Args:
        value: Value to convert (BaseModel, Enum, dict, list, or primitive)

    Returns:
        JSON-serializable representation of the value
    """
    if isinstance(value, BaseModel):
        return {k: _convert_to_serializable(v) for k, v in value.__dict__.items()}
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [_convert_to_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_to_serializable(v) for k, v in value.items()}
    else:
        return value


def serialize_metadata(metadata: PokemonMetadata) -> Dict[str, Any]:
    """
    Convert BAML metadata to JSON-serializable dictionary.

    Args:
        metadata: Pydantic model from BAML extraction

    Returns:
        Dictionary with all nested objects converted to primitives
    """
    return _convert_to_serializable(metadata)  # type: ignore


def process_single_image(image_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Extract metadata from a single Pokemon image using BAML.

    Args:
        image_path: Path to Pokemon image file
        verbose: Enable detailed logging

    Returns:
        Extracted metadata dictionary

    Raises:
        Exception: If metadata extraction fails
    """
    if verbose:
        print(f"Processing: {image_path}")

    pokemon_name = image_path.parent.name

    with open(image_path, "rb") as f:
        image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode('utf-8')

    pokemon_image = Image.from_base64("image/png", image_b64)
    metadata = b.ExtractPokemonMetadata(img=pokemon_image, pokemon_name=pokemon_name)
    metadata_dict = serialize_metadata(metadata)

    try:
        project_root = Path(__file__).parent
        relative_path = image_path.relative_to(project_root)
        metadata_dict['image_path'] = str(relative_path)
    except ValueError:
        metadata_dict['image_path'] = str(image_path)

    metadata_dict['pokemon_name'] = pokemon_name

    pokemon_name_lower = pokemon_name.lower()
    image_stem = image_path.stem
    output_path = image_path.parent / f"{pokemon_name_lower}_{image_stem}.json"

    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    if verbose:
        print(f"  Saved: {output_path.name}")

    return metadata_dict


def process_all_pokemon(
    images_dir: Optional[Path] = None,
    verbose: bool = True,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Process all Pokemon images and generate metadata files.

    Args:
        images_dir: Path to images directory (defaults to ./images)
        verbose: Enable detailed progress output
        skip_existing: Skip images with existing metadata files

    Returns:
        Statistics dictionary with processing results

    Raises:
        FileNotFoundError: If images directory doesn't exist
    """
    if images_dir is None:
        images_dir = Path(__file__).parent / "images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    print("="*80)
    print("POKEMON METADATA BATCH PROCESSOR")
    print("="*80)
    print(f"Source directory: {images_dir}")
    print(f"Skip existing: {skip_existing}")
    print("="*80)

    pokemon_folders = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"\nDiscovered {len(pokemon_folders)} Pokemon species")

    stats = {
        "total_images": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": []
    }

    for pokemon_folder in pokemon_folders:
        pokemon_name = pokemon_folder.name
        image_files = sorted(pokemon_folder.glob("*.png"))

        if not image_files:
            if verbose:
                print(f"\n{pokemon_name}: No images found")
            continue

        if verbose:
            print(f"\n{pokemon_name}: {len(image_files)} images")

        stats["total_images"] += len(image_files)

        for image_path in image_files:
            pokemon_name_lower = pokemon_name.lower()
            image_stem = image_path.stem
            metadata_path = image_path.parent / f"{pokemon_name_lower}_{image_stem}.json"

            if skip_existing and metadata_path.exists():
                if verbose:
                    print(f"  Skipped: {image_path.name} (exists)")
                stats["skipped"] += 1
                continue

            try:
                process_single_image(image_path, verbose=verbose)
                stats["processed"] += 1

            except Exception as e:
                stats["failed"] += 1
                error_str = str(e)

                if "content_filter" in error_str or "ResponsibleAIPolicyViolation" in error_str:
                    error_msg = f"{pokemon_name}/{image_path.name}: Content filter triggered"
                    if verbose:
                        print(f"  Blocked: {image_path.name} (content filter)")
                else:
                    error_msg = f"{pokemon_name}/{image_path.name}: {error_str[:100]}"
                    if verbose:
                        print(f"  Failed: {image_path.name} - {error_str[:80]}")

                stats["errors"].append(error_msg)

    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total: {stats['total_images']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

    if stats["errors"]:
        content_filter_errors = [e for e in stats["errors"] if "Content filter" in e]
        other_errors = [e for e in stats["errors"] if "Content filter" not in e]

        if content_filter_errors:
            print(f"\nContent Filter Blocks: {len(content_filter_errors)}")
            for error in content_filter_errors[:10]:
                pokemon_name = error.split('/')[0]
                print(f"  - {pokemon_name}")
            if len(content_filter_errors) > 10:
                print(f"  ... {len(content_filter_errors) - 10} more")

        if other_errors:
            print(f"\nProcessing Errors: {len(other_errors)}")
            for error in other_errors[:10]:
                print(f"  - {error}")
            if len(other_errors) > 10:
                print(f"  ... {len(other_errors) - 10} more")

    print("="*80)

    return stats


def verify_metadata_structure(metadata: PokemonMetadata) -> None:
    """
    Verify extracted metadata structure and display sample information.

    Args:
        metadata: Metadata object from BAML extraction
    """
    print("\n" + "="*80)
    print("METADATA STRUCTURE VERIFICATION")
    print("="*80)

    required_fields = [
        "prompts",
        "visual_attributes",
        "categorical_attributes",
        "semantic_attributes"
    ]

    metadata_dict = serialize_metadata(metadata)

    print("\nRequired fields:")
    for field in required_fields:
        exists = field in metadata_dict
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {field}")

    if "prompts" in metadata_dict:
        num_prompts = len(metadata_dict["prompts"])
        print(f"\nPrompts: {num_prompts} generated")
        if 5 <= num_prompts <= 7:
            print("  Valid range (5-7)")
        else:
            print(f"  Warning: Expected 5-7, got {num_prompts}")

        if metadata_dict["prompts"]:
            print(f"\nSample: \"{metadata_dict['prompts'][0]}\"")

    if "visual_attributes" in metadata_dict:
        va = metadata_dict["visual_attributes"]
        print(f"\nColors: {va.get('primary_colors', [])}")
        print(f"Dominant: {va.get('color_distribution', {}).get('primary', 'N/A')}")

    if "categorical_attributes" in metadata_dict:
        ca = metadata_dict["categorical_attributes"]
        type2 = ca.get('type2', 'None')
        print(f"\nType 1: {ca.get('type1', 'N/A')}")
        print(f"Type 2: {type2 if type2 else 'None'}")
        print(f"Generation: {ca.get('generation', 'N/A')}")

    print("\n" + "="*80)


def main() -> None:
    """Main entry point for metadata generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Pokemon metadata using BAML vision API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Path to images directory"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess images with existing metadata"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    stats = process_all_pokemon(
        images_dir=images_dir,
        verbose=not args.quiet,
        skip_existing=not args.no_skip_existing
    )

    exit(1 if stats["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
