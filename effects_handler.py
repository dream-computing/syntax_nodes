import cv2
import numpy as np
  
# First version of apply method
def apply_v1(image, puzzle_image, puzzle_mask):
  return puzzle_image, puzzle_mask

# Input: The image, image with the puzzle and the puzzle mask
# Output: The puzzle image with shadow effect applied and the puzzle mask
def apply_relief_and_shadow(puzzle_image, puzzle_mask):
  # Create and apply emboss filter
  kernel = np.zeros((3,3))
  kernel[0,1] = -1
  kernel[2,1] = 1
  kernel[1, 0] = -1
  kernel[1, 2] = 1

  effect_mask = cv2.filter2D(puzzle_mask, -1, kernel) 

  # Invert the mask
  effect_mask = 255 - effect_mask

  # Apply the mask
  puzzle_image = cv2.bitwise_or(puzzle_image, puzzle_image, mask = effect_mask)

  # Apply a small blur to soften the edges
  puzzle_image = cv2.GaussianBlur(puzzle_image, (3,3), sigmaX = 1.5, sigmaY = 1.5)
  puzzle_image = puzzle_image.astype(np.uint8)

  # Return the result
  return puzzle_image, puzzle_mask

#Input: The background image and the foreground image (both must have the same shape)
#Output: The original image with a background
def add_background(background, foreground, alpha_mask):

  # Check if images have the same shape
  try:
    assert (background.shape == foreground.shape), 'Images must have same size and same number of channels.'
  except AssertionError as err:
    print('Error: {}'.format(err))
    exit(1)

  # Check if images have the same data type
  try:
    assert (background.dtype == foreground.dtype), 'Images must have same data type.'
  except AssertionError as err:
    print('Error: {}'.format(err))
    exit(1)

  # apply mask to foreground image
  foreground = cv2.multiply(alpha_mask, foreground)

  # apply negative mask to background image
  background = cv2.multiply(1 - alpha_mask, background)

  # add background and foreground images
  blended = cv2.add(background, foreground)

  # return blended image
  return blended
