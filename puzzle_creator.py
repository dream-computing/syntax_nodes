import cv2
import numpy as np

# Receives the size of the desired mask and the piece size
# Returns the puzzle mask
def create_puzzle_mask(height, width, piece_size, thickness):
  puzzle_mask = np.zeros((height, width), dtype=np.uint8)
  # Draw a bounding box on the puzzle mask
  cv2.rectangle(puzzle_mask, (0,0), (puzzle_mask.shape[1]-1, puzzle_mask.shape[0]-1), 255, thickness=thickness)

  # Loop through each square piece
  for i in range(0, puzzle_mask.shape[0], piece_size):
    for j in range(0, puzzle_mask.shape[1], piece_size):
      
      # Check if it is not the upper border before drawing
      if i != 0:

        # Declare useful variables
        mid_j = (piece_size//2) # 1/2 of piece
        l_j = (piece_size//3) # 1/3 of piece
        r_j = (2*piece_size)//3 # 2/3 of piece
        offset = 0 # piece joint offset
        half_joint_size = (piece_size//16) # half height of piece joint
        
        # Draw borders of piece
        if (j//piece_size)%2:
          cv2.ellipse(puzzle_mask, (j + l_j, i), (l_j, 2), 0, 180, 280, 255, thickness=thickness, lineType = cv2.LINE_8)
          cv2.ellipse(puzzle_mask, (j + r_j, i), (piece_size - r_j, 2), 0, 260, 360, 255, thickness=thickness, lineType = cv2.LINE_8)
          offset = -2
        else:
          cv2.ellipse(puzzle_mask, (j + l_j, i), (l_j, 2), 0, 80, 180, 255, thickness=thickness, lineType = cv2.LINE_8)
          cv2.ellipse(puzzle_mask, (j + r_j, i), (piece_size - r_j, 2), 0, 0, 100, 255, thickness=thickness, lineType = cv2.LINE_8)
          offset = 2

        # Select the piece hole direction
        if np.random.randint(1,3) == 1:
          # Draw upwards  
          cv2.ellipse(puzzle_mask, (j + mid_j, i + offset - 2*half_joint_size), ((r_j - l_j)//2, (5*half_joint_size)//2), 90, 40, 320, 255, thickness=thickness, lineType = cv2.LINE_8)

        else:
          # Draw downwards
          cv2.ellipse(puzzle_mask, (j + mid_j, i + offset + 2*half_joint_size), ((r_j - l_j)//2, (5*half_joint_size)//2), 270, 40, 320, 255, thickness=thickness, lineType = cv2.LINE_8)


      # Check if it is not the left border before drawing
      if j != 0:

        # Declare useful variables
        mid_i = (piece_size//2) # 1/2 of piece
        l_i = (piece_size//3) # 1/3 of piece
        r_i = (2*piece_size)//3 # 2/3 of piece
        offset = 0 # piece joint offset
        half_joint_size = (piece_size//16) # half height of piece joint

        # Draw borders of piece
        if (i//piece_size)%2:
          cv2.ellipse(puzzle_mask, (j, i + l_i), (2, l_i), 0, 270, 370, 255, thickness=thickness, lineType = cv2.LINE_8)
          cv2.ellipse(puzzle_mask, (j, i + r_i), (2, l_i), 0, -10, 90, 255, thickness=thickness, lineType = cv2.LINE_8)
          offset = 2
        else:
          cv2.ellipse(puzzle_mask, (j, i + l_i), (2, l_i), 0, 170, 270, 255, thickness=thickness, lineType = cv2.LINE_8)
          cv2.ellipse(puzzle_mask, (j, i + r_i), (2, l_i), 0, 90, 190, 255, thickness=thickness, lineType = cv2.LINE_8)
          offset = -2
          
        # Select the piece hole direction
        if np.random.randint(1,3) == 1:
          # Draw to the left
          cv2.ellipse(puzzle_mask, (j + offset - 2*half_joint_size, i + mid_i), ((5*half_joint_size)//2, (r_i - l_i)//2), 0, 40, 320, 255, thickness=thickness, lineType = cv2.LINE_8)

        else:
          # Draw to the right
          cv2.ellipse(puzzle_mask, (j + offset + 2*half_joint_size, i + mid_i), ((5*half_joint_size)//2, (r_i - l_i)//2), 180, 40, 320, 255, thickness=thickness, lineType = cv2.LINE_8)

  # Return the puzzle mask
  return puzzle_mask

# Receives the image and the type of puzzle desired
# Returns the image with the puzzle drawn and the puzzle mask
def create(image, piece_size):

  # Adjust the image size to make all pieces have the same size
  row_leftover = image.shape[0]%piece_size
  col_leftover = image.shape[1]%piece_size
  image = cv2.resize(image, (image.shape[1] - col_leftover, image.shape[0] - row_leftover))

  # Create the puzzle based on piece size
  puzzle_mask = create_puzzle_mask(image.shape[0], image.shape[1], piece_size = piece_size, thickness = 2)

  # Draw the puzzle on the image
  puzzle_image = image.copy()
  puzzle_image = cv2.bitwise_or(puzzle_image, np.dstack([puzzle_mask]*3))

  puzzle_mask = np.expand_dims(puzzle_mask, axis=2)

  # Return the puzzle image and mask
  return puzzle_image, puzzle_mask