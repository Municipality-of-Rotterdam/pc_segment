"""Helper functions."""

import re


def get_collection_info(pc_filename: str) -> str:
    """Extract the *collection code* from a point cloud filename.

    This function parses filenames following the convention used for
    Cyclomedia or similar point cloud datasets, where file paths encode
    recording metadata, for example:

        nl-rott-230412-7415-laz/filtered_1647_8639.laz
        → "nl-rott-230412-7415-laz"

    It captures the part between the prefix `"nl-rott-"` and the suffix `"-laz"`,
    then reconstructs the full collection code.

    Parameters
    ----------
    pc_filename : str
        The full point cloud filename or path containing the collection info.

    Returns:
    -------
    str
        The reconstructed collection code, e.g. `"nl-rott-230412-7415-laz"`.

    Raises:
    ------
    ValueError
        If the expected pattern cannot be found in the filename.

    Notes:
    -----
    - The function assumes that the filename includes both "nl-rott-" and "-laz".
    - This metadata is used to group output tiles and maintain consistent directory structure.
    """
    prefix = "nl-rott"
    suffix = "laz"

    # Regex pattern capturing everything between prefix and suffix
    pattern = rf"{prefix}-(.*?)-{suffix}"

    match = re.search(pattern, pc_filename)
    if match:
        captured_part = match.group(1)
        return f"{prefix}-{captured_part}-{suffix}"

    err_msg = f"Cannot determine collection info of input pointcloud {pc_filename}."
    raise ValueError(err_msg)
