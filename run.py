import cv2
import argparse
import numpy as np 
import os
import modules.puzzle_creation as puzzle_creation
import modules.effects as effects
import modules.transformations as transformations

def main():

  # Define and parse the arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('-b', '--background_path', dest='background_path', metavar='', type=str, default=None, help='Path to the background image. Default is white background')
  parser.add_argument('-i', '--input_path', dest='input_path', metavar='', type=str, default='examples/lena.png', help='Path to the input image. Default is \'examples/lena.png\'')
  parser.add_argument('-n', '--n_moving_pieces', dest='n_moving_pieces', metavar='', type=int, default=3, help='Number of pieces that are going to be moved. If the number given is bigger than the number of pieces, the default behaviour is to assume n_moving_pieces = number of pieces in the puzzle. Default number is 3.')
  parser.add_argument('-o', '--output_path', dest='output_path', metavar='', type=str, default='output.png', help='Path to save the output image (directory and name). Make sure that the directory exists. Default is \'output.png\'.')
  parser.add_argument('-p', '--piece_size', dest='piece_size', metavar='', type=int, default=64, help='Size of the puzzle pieces. This number must be bigger or equal than 32. Default is 64.')
  parser.add_argument('-s', '--select_pieces', action='store_true', dest='select_pieces', default=False, help='Selection of which pieces should be removed. Enter \'True\' to select. Default is \'False\'.')
  parser.add_argument('-r', '--random_seed', dest = 'random_seed', metavar='', type=int, default=-1, help='Seed for the random generator. The number must be greater than zero. Default does not fix a seed.')
  args = parser.parse_args()

  # Read the image
  original_image = cv2.imread(args.input_path)

  # Read background image
  background_image = cv2.imread(args.background_path)

  # Declare the piece size (to be changed later)
  piece_size = -1

  # Set the seed if passed by argument
  if(args.random_seed > 0):
    np.random.seed(args.random_seed)

  # Asserts
  try:
    # Check if image exists
    assert (original_image is not None), 'Image does not exist.'

    # Check if the piece type exists and get the piece size
    assert (args.piece_size >= 32), 'Piece size must be greater or equal than 32.'

    # Check if image size is greater than two times the piece size
    assert ((original_image.shape[0] >= 2*args.piece_size and original_image.shape[1] >= args.piece_size) or (original_image.shape[0] >= args.piece_size and original_image.shape[1] >= 2*args.piece_size)), 'Input image has smaller dimensions than the puzzle piece. Make sure that the image can fit at least two puzzle pieces.'

    # Check if number of moved pieces is greater or equal than zero
    assert (args.n_moving_pieces >= 0), 'Number of pieces to be moved needs to be greater or equal than zero.'

  except AssertionError as err:
    print('Error: {}'.format(err))
    exit(1)
  
  # Create the puzzle mask and the puzzle image
  puzzle_image, puzzle_mask = puzzle_creation.create(original_image, args.piece_size)

  # If the user does not give a correct path to the background, create a white background image
  if background_image is None:
    background_image = np.full(puzzle_image.shape, (255, 255, 255), dtype=np.uint8)

  # Assert that the background has dimensions greater or equal than the image
  try:
    assert (background_image.shape[0] >= puzzle_image.shape[0] and background_image.shape[1] >= puzzle_image.shape[1]), 'Background needs to be bigger or equal than the image.'
  except AssertionError as err:
    print('Error: {}'.format(err))
    exit(1)

  # Transform image
  puzzle_image, puzzle_mask, foreground_mask = transformations.transform_v1(puzzle_image, puzzle_mask, args.piece_size, background_image.shape, args.n_moving_pieces, args.select_pieces)

  # Add background to image
  puzzle_image = effects.add_background(background_image, puzzle_image, foreground_mask)
  
  # Apply relief and shadow effects to the image
  puzzle_image, puzzle_mask = effects.apply_relief_and_shadow(puzzle_image, puzzle_mask)

  # Save the output image and the mask
  cv2.imwrite(args.output_path, puzzle_image)

if __name__ == '__main__':
  main()