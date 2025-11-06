import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
import tempfile
import json
from tempfile import NamedTemporaryFile
import itertools

import matplotlib.pyplot as plt
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
import cv2
from segment_anything import SamPredictor

from pc_segment.segment_pc import (
    prompt_segment,
    initialize_model,
    combine_masks,
    segment_pointcloud_img,
    show_points, 
    save_image_prompt, 
    prepare_image
)



def test_show_points():
    fig, ax = plt.subplots()
    coords = np.array([[10, 20], [30, 40], [50, 60]])
    show_points(coords, ax)
    assert len(ax.collections) > 0  # Check if points are plotted


def test_save_image_prompt():
    # Create a blank image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    with NamedTemporaryFile(suffix=".png", delete=True) as temp_img, NamedTemporaryFile(suffix=".png", delete=True) as temp_output:
        cv2.imwrite(temp_img.name, img)
        save_image_prompt(temp_img.name, [(10, 10), (20, 20), (30, 30)], temp_output.name)
        output_img = cv2.imread(temp_output.name)
        assert output_img is not None and output_img.shape[0] > 0  # Check if file was created


def test_prepare_image():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    transform = ResizeLongestSide(128)
    device = torch.device("cpu")
    processed_img = prepare_image(img, transform, device)
    assert isinstance(processed_img, torch.Tensor)
    assert processed_img.shape[0] == 3  # Check if converted to channel-first format

# Test for initialize_model function
def test_initialize_model():
    """Test the initialize_model function."""
    model_path = "/path/to/model.pth"  # Path to the model (this won't be used in the test due to mocking)
    mock_model = mock.Mock()

    # Mock the registry so that when "vit_h" is accessed, it returns a mock model
    with mock.patch.dict("segment_anything.sam_model_registry", {"vit_h": mock.Mock()}) as mock_registry:

        # Ensure that the 'to' method works correctly (we don't care about its functionality here, just that it's called)
        mock_model.to.return_value = mock_model  # This ensures chaining .to() works
        
        # Mock the model instantiation with the checkpoint argument
        mock_registry["vit_h"].return_value = mock_model
        
        # Call the function that uses the mock registry and predictor
        predictor = initialize_model(model_path)
        
        # Check that the model was correctly initialized and is an instance of SamPredictor
        assert isinstance(predictor, SamPredictor)
        
        # Ensure that the registry was accessed with the correct model type and checkpoint path
        mock_registry["vit_h"].assert_called_once_with(checkpoint=model_path)
        
        # Ensure that the .to() method was called once with the appropriate device ("cpu" or "cuda")
        mock_model.to.assert_called_once_with(device="cpu")  # Since test agent doesn't have CUDA


# Test for checking if the correct device ('cpu' or 'cuda') is passed when CUDA is available
def test_initialize_model_with_cuda_available():
    """Test that when CUDA is available, the model is initialized on the CUDA device."""
    model_path = "/path/to/model.pth"
    mock_model = mock.Mock()

    with mock.patch("torch.cuda.is_available", return_value=True):
        # Use MagicMock for the registry to support subscripting (i.e., dictionary-like access)

        # Mock the registry so that when "vit_h" is accessed, it returns a mock model
        with mock.patch.dict("segment_anything.sam_model_registry", {"vit_h": mock.Mock()}) as mock_registry:
            
            # Ensure that the 'to' method works correctly (we don't care about its functionality here, just that it's called)
            mock_model.to.return_value = mock_model  # This ensures chaining .to() works
            
            # Mock the model instantiation with the checkpoint argument
            mock_registry["vit_h"].return_value = mock_model
            
            # Call the function that uses the mock registry and predictor
            predictor = initialize_model(model_path)
            
            # Check that the model was correctly initialized and is an instance of SamPredictor
            assert isinstance(predictor, SamPredictor)
            
            # Ensure that the registry was accessed with the correct model type and checkpoint path
            mock_registry["vit_h"].assert_called_once_with(checkpoint=model_path)
            
            # Ensure that the .to() method was called once with the appropriate device ("cpu" or "cuda")
            mock_model.to.assert_called_once_with(device="cuda")


def test_initialize_model_invalid_checkpoint():
    """Test that the model raises a FileNotFoundError when an invalid checkpoint path is provided."""
    
    invalid_model_path = "/path/to/nonexistent/model.pth"
    
    # Assert that a FileNotFoundError is raised for an invalid model path
    with pytest.raises(FileNotFoundError):
        initialize_model(invalid_model_path)
            
def test_initialize_model_valid_checkpoint():
    """Test that the model raises a EOFError when a valid but empty path is provided."""
    
    # Create a temporary file with the .pth extension
    with tempfile.NamedTemporaryFile(delete=True, suffix='.pth') as temp_file:
        valid_model_path = temp_file.name
    
        # Assert that the correct exception is raised for a valid but empty model path
        with pytest.raises(EOFError):
            initialize_model(valid_model_path)

            
# Mocking the predictor function for more comprehensive test coverage
@pytest.fixture
def mock_predictor():
    """Fixture to mock the SamPredictor."""
    mock_predictor = mock.Mock(spec=SamPredictor)
    mock_predictor.device = "cpu"
    
    mock_predictor.set_image = mock.Mock()
    
    # First, mock the 'transform' attribute as a mock object
    mock_predictor.transform = mock.Mock()
    
    # Now, mock the 'apply_coords_torch' method on the 'transform' attribute
    mock_predictor.transform.apply_coords_torch = mock.Mock(return_value=torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
    # Mocking the predict_torch method
    mock_predictor.predict_torch = mock.Mock(
        side_effect=[
            (torch.ones(4, 1, 4, 4), None, None),  # First call returns (4 masks, batch size 1, 4x4 image)
            (torch.zeros(3, 1, 4, 4), None, None),  # Second call returns (3 masks, batch size 1, 4x4 image)
        ]
    )
    return mock_predictor

def test_combine_masks_with_mock_predictor(mock_predictor):
    """Test the combine_masks function with mocked predictor."""
    
    # Test case where we expect to squeeze (1 mask)
    masks, _, _ = mock_predictor.predict_torch(
        point_coords=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        point_labels=torch.tensor([1, 2]),
        boxes=None,
        multimask_output=False,
    )
    
    # Apply combine_masks on the generated masks
    mask_img = combine_masks(masks)
    
    # Assert that the output matches expected values
    expected_mask = np.ones((4, 4))
    np.testing.assert_array_equal(mask_img, expected_mask)

    # Test case where we expect not to squeeze (3 masks)
    masks, _, _ = mock_predictor.predict_torch(
        point_coords=torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        point_labels=torch.tensor([1, 2]),
        boxes=None,
        multimask_output=False,
    )
    
    # Apply combine_masks on the generated masks
    mask_img = combine_masks(masks)
    
    # Assert that the output matches expected values for 3 masks
    expected_mask = np.zeros((4,4))
    
    np.testing.assert_array_equal(mask_img, expected_mask)


# Parametrize over num_masks, height (H), and width (W)
@pytest.mark.parametrize(
    "num_masks, H, W", [
        (1, 1016, 958),  # Case 1: 1 mask with a standard image size
        (5, 1016, 958),  # Case 2: 5 masks with a standard image size
        (10, 512, 512),  # Case 3: 10 masks with smaller image size
        (14, 256, 256),  # Case 4: 14 masks with even smaller image size
        (3, 256, 512),   # Case 5: 3 masks with rectangular image
        (3, 1024, 2048), # Case 6: 3 masks with a large, wide image
    ]
)
def test_combine_masks_parametrized(num_masks, H, W):
    # Calculate the width of each vertical bin (with an extra bin for the last column)
    bin_width = W // (num_masks + 1)

    # Create a tensor for the masks with shape (num_masks, 1, H, W)
    masks = torch.zeros(num_masks, 1, H, W)  # Initialize tensor of zeros (no mask)

    # Generate each mask by setting True values in the appropriate bins
    for i in range(num_masks):
        # Set the pixels in the first (i+1) bins to True
        masks[i, 0, :, : (bin_width * (i + 1))] = 1  # Set True in the first (i+1) bins
    
    # Call combine_masks on the tensor
    result = combine_masks(masks)
    
    # Create the expected result:
    # - The first bin (leftmost) should have all ones.
    # - The second bin should have all twos, and so on.
    # - The last bin (rightmost) should have zeros.
    
    expected_result = np.zeros((H, W), dtype=int)
    
    # Fill in each bin's expected value (1 through N)
    for i in range(num_masks):
        expected_result[:, bin_width * (i): bin_width * (i + 1)] = i + 1

    # Assert that the combined result matches the expected outcome
    np.testing.assert_array_equal(result, expected_result)

# Test case for the prompt_segment function
@patch('cv2.imread')  # Mock cv2.imread
@patch('cv2.imwrite')  # Mock cv2.imwrite
def test_prompt_segment(mock_imwrite, mock_imread):
    # Test prompt_segment with one image (i.e. batch of one image)
    # Setup mock image reading
    mock_imread.return_value = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    mock_imwrite.return_value = True

    # Mock predictor and its methods
    predictor = MagicMock()
    predictor.device = torch.device('cpu')  # Use CPU for simplicity
    predictor.set_image = MagicMock()
    predictor.transform.apply_coords_torch = MagicMock(return_value=torch.tensor([[0, 0], [1, 1]]))
    predictor.predict_torch = MagicMock()

    # Test data (Batch size = 1)
    image_paths = ["fake_image_path.jpg"] # batch size 1
    point_prompts = [[[0, 0], [1, 1]]]  # Two points, batch size 1
    point_labels = [[[1], [2]]]  # Two labels corresponding to the points, batch size 1
    outfnames = ["output_mask.jpg"]

    # Mock the output of predictor.model (batch masks) (output masks with shape C=num_masks, B=1, H=2, W=2)
    dummy_masks = [
        {"masks": torch.tensor([[[[True, False], [False, True]]],  # Mask 1
                                 [[[False, True], [True, False]]]],  # Mask 2
                               dtype=torch.bool)}
    ]
    expected_result = [np.array([[1, 2], [2, 1]])]

    # Mock the model call (which replaced predict_torch)
    predictor.model.return_value = dummy_masks
    # Call the function to test
    result = prompt_segment(predictor, image_paths, point_prompts, point_labels, outfnames)

    # Assertions
    # Assert the image was read correctly
    mock_imread.assert_called_once_with(image_paths[0])
    
    # Assert the result is a list of numpy arrays
    assert isinstance(result, list)
    assert all(isinstance(mask, np.ndarray) for mask in result)

    # Assert the output mask is as expected (combination of dummy_masks)
    np.testing.assert_array_equal(result[0], expected_result[0])

    # Ensure predictor.model() was called once with expected batched input
    predictor.model.assert_called_once()

    # First, check that cv2.imwrite was called exactly once (per input image)
    mock_imwrite.assert_called_once()

    # Retrieve the arguments passed to the mock (call_args returns a tuple)
    actual_args, _ = mock_imwrite.call_args

    # Ensure the filename matches
    assert actual_args[0] == outfnames[0], f"Expected filename {outfnames[0]}, but got {actual_args[0]}"

    # Ensure the mask (second argument) is correct
    assert np.array_equal(actual_args[1], expected_result[0]), f"Expected result:\n{expected_result}\nBut got:\n{actual_args[1]}"


@pytest.fixture
def mock_args():
    """Fixture to mock argparse.Namespace for segment_pointcloud_img."""
    mock_args = MagicMock()
    mock_args.img_metadata = "mock_processed_files.json"
    mock_args.model_path = "/path/to/mock_model.pth"
    mock_args.segment_dir = "mock_output_directory"
    mock_args.img_dir = "mock_image_directory"
    mock_args.segment_metadata = "mock_segmented_files.json"
    mock_args.batch_size = 1
    return mock_args

@patch("os.makedirs")  # Mock os.makedirs
@patch("builtins.open", new_callable=mock.mock_open)  # Mock open() for JSON files
@patch("json.dump")  # Mock json.dump to avoid file writing
@patch("pc_segment.segment_pc.initialize_model")  # Mock the initialize_model function
@patch("pc_segment.segment_pc.prompt_segment")  # Mock the prompt_segment function
@patch("cv2.imread")  # Mock cv2.imread to return a valid image
@patch("matplotlib.pyplot.savefig")  # Mock plt.savefig to prevent actual file writing
def test_segment_pointcloud_img(
    mock_savefig,
    mock_imread,
    mock_prompt_segment,
    mock_initialize_model,
    mock_json_dump,
    mock_open,
    mock_makedirs,
    mock_args,
):
    """Test the segment_pointcloud_img function."""

    # Create a temporary image file that `cv2.imread` can successfully read
    with NamedTemporaryFile(suffix=".png", delete=True) as temp_img:
        temp_img_path = temp_img.name

        # Create a blank white image and save it
        dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        cv2.imwrite(temp_img_path, dummy_image)

        # Ensure `cv2.imread()` returns the dummy image
        mock_imread.return_value = dummy_image

        # Mock the processed files JSON content
        img_metadata_dict = {
            "nl-rott-230420-7415-laz/las_processor_bundled_out/filtered_1842_8767.laz": {
                "img_path": temp_img_path,
                "prompt_path": "mock_prompt.json",
            },
        }
        
        prompt_dict = {
            "point_coords": [[0, 0], [1, 1]],
            "point_labels": [[1], [2]],
        }
        
        # Use itertools.chain to return img_metadata_dict once, then prompt_dict for all subsequent calls
        mock_open.return_value.__enter__.return_value.read.side_effect = itertools.chain(
            [json.dumps(img_metadata_dict)],  # First call returns img_metadata_dict
            itertools.repeat(json.dumps(prompt_dict))  # All later calls return prompt_dict
        )
        
        # Mock the return value of initialize_model to return a mock predictor
        mock_predictor = mock.Mock(spec=SamPredictor)
        mock_predictor.device = torch.device("cpu")
        mock_initialize_model.return_value = mock_predictor

        # Mock the return value of prompt_segment
        mock_prompt_segment.return_value = [np.ones((256, 256))]  # Dummy mask

        # Call the function
        result = segment_pointcloud_img(mock_args)

        # Ensure result is a dictionary and contains the expected key
        assert isinstance(result, dict)
        assert "nl-rott-230420-7415-laz/las_processor_bundled_out/filtered_1842_8767.laz" in result
        assert result["nl-rott-230420-7415-laz/las_processor_bundled_out/filtered_1842_8767.laz"].endswith("pred_mask.png")

        # Ensure `cv2.imread()` was called with the correct image path
        mock_imread.assert_called_with(temp_img_path)

        # Ensure `save_image_prompt` was called by checking that `plt.savefig` was used
        mock_savefig.assert_called_once()

        # Ensure `prompt_segment` was called once with the correct parameters
        mock_prompt_segment.assert_called_once()

        # Ensure json.dump was called to save the segmented metadata
        mock_json_dump.assert_called_once()

        # Ensure open was called for reading metadata and writing the segment metadata file
        mock_open.assert_any_call(mock_args.img_metadata)  # For reading processed file JSON
        mock_open.assert_any_call(mock_args.segment_metadata, "w")  # For writing the output JSON

