import cv2
import numpy as np
import queue


# Input
def transform_v1(puzzle_image, puzzle_mask, piece_size, background_shape, n_moving_pieces, select_pieces):

  # Create visited matrix
  vis = np.zeros(puzzle_mask.shape)

  # Create offset lists
  dy = [-1,  0, 0, 1]
  dx = [ 0, -1, 1, 0]

  # Create variables for rows and cols
  rows = puzzle_image.shape[0]
  cols = puzzle_image.shape[1]

  # Create foreground_mask, where 1 is foreground and 0 background
  foreground_mask = np.full(puzzle_mask.shape, 1, dtype=np.uint8)

  # Initialize variables
  puzzle_i = 0
  puzzle_j = 0

  # Check if background and original shapes are different
  if background_shape != puzzle_image.shape:
    puzzle_image, puzzle_i, puzzle_j = add_padding(puzzle_image, background_shape)
    puzzle_mask, _, _ = add_padding(puzzle_mask, background_shape)
    foreground_mask, _, _ = add_padding(foreground_mask, background_shape)

  # create array with pieces indices
  indices = np.indices((rows // piece_size, cols // piece_size)).transpose((1, 2, 0))
  indices = indices.reshape(((rows // piece_size * cols // piece_size), 2))
  if(select_pieces == False):
    indices = np.random.permutation(indices)
  
  # if n_moving_pieces is bigger than the number of pieces
  # n_moving_pices = number of pieces
  n_moving_pieces = np.minimum(len(indices), n_moving_pieces)
  
  if(select_pieces == True):
    order = piece_selection(indices, puzzle_image, piece_size, puzzle_j, puzzle_i, n_moving_pieces)

  # Generate all possible positions for translate the pieces and permutate them
  transformations = make_transformations(piece_size, puzzle_i, puzzle_j, rows, cols)
  transformations = np.random.permutation(transformations)

  # iterate over pieces indices 
  for ind in range(0, n_moving_pieces):
    
    # get image coordinate from index
    if(select_pieces == False):
      i, j = index_to_coordinate(indices[ind], piece_size)
    else:
      i, j = index_to_coordinate(indices[order[ind]], piece_size)
    
    # add image offset to coordinates
    i += puzzle_i
    j += puzzle_j

    # Get the translation offset (ty,tx) and rotation angle in degrees for this piece
    # If there's no space left in background, the piece is just deleted
    ty = 0
    tx = 0
    rotation_angle = 0
    if ind < len(transformations):
      rotation_angle = np.random.randint(0, 361)
      offy, offx = transformations[ind]
      ty = - (i - puzzle_i) + offy
      tx = - (j - puzzle_j) + offx
    cy = i + ty
    cx = j + tx
    

    # Do a BFS to visit each pixel of the piece with center at (i, j)
    q = queue.Queue(maxsize = 0)
    q.put((i, j))
    vis[i - puzzle_i, j - puzzle_j] = True

    while not q.empty():
      y, x = q.get()

      # Apply the translation and rotation to the piece pixel before deleting it
      newy = y + ty
      newx = x + tx
      newy, newx = rotate_around((cy,cx), (newy, newx), rotation_angle)
      if(0 <= newy < puzzle_image.shape[0] and 0 <= newx < puzzle_image.shape[1]):
        puzzle_image[newy, newx] = puzzle_image[y,x]
        puzzle_mask[newy, newx] = puzzle_mask[y,x]
        foreground_mask[newy, newx] = foreground_mask[y,x]

      # Turn each pixel from the piece with center at (i, j) black
      puzzle_image[y, x] = 0
      foreground_mask[y, x] = 0

      # If it's a border, delete its neighbors and mark them as visited 
      if puzzle_mask[y,x] == 255:
        puzzle_mask[y,x] = 0
        foreground_mask[y, x] = 0
        vy = [0, 1, 0, -1, 1, -1, 1, -1]
        vx = [1, 0, -1, 0, 1, 1, -1, -1]
        for k in range(0, 8):
          if puzzle_i <= y + vy[k] < rows + puzzle_i and puzzle_j <= x + vx[k] < cols + puzzle_j and vis[y + vy[k] - puzzle_i, x + vx[k] - puzzle_j] == False:
            # Apply the translation and rotation to the piece at the border before deleting it
            newy = y + vy[k] + ty
            newx = x + vx[k] + tx
            newy, newx = rotate_around((cy,cx), (newy, newx), rotation_angle)
            if(0 <= newy < puzzle_image.shape[0] and 0 <= newx < puzzle_image.shape[1]):
              puzzle_image[newy, newx] = puzzle_image[y + vy[k], x + vx[k]]
              puzzle_mask[newy, newx] = puzzle_mask[y + vy[k], x + vx[k]]
              foreground_mask[newy, newx] = foreground_mask[y + vy[k], x + vx[k]]
            
            vis[y + vy[k] - puzzle_i, x + vx[k] - puzzle_j] = True
            puzzle_mask[y + vy[k], x + vx[k]] = 0
            puzzle_image[y + vy[k], x + vx[k]] = 0
            foreground_mask[y + vy[k], x + vx[k]] = 0
        continue

      # Iterate over adjacents
      for k in range(0, 4):
        if puzzle_i <= y + dy[k] < rows + puzzle_i and 0 <= x + dx[k] < cols + puzzle_j and vis[y + dy[k] - puzzle_i, x + dx[k] - puzzle_j] == False:
          vis[y + dy[k] - puzzle_i, x + dx[k] - puzzle_j] = True
          q.put((y + dy[k], x + dx[k]))
  
  # Filter similar to erosion to fix some borders imperfections when pieces are deleted
  for iter in range(2):
    kernel = np.ones((9,9), dtype=np.int16)
    kernel[4,4] = 0
    filtered_mask = puzzle_mask.astype(np.int16)
    filtered_mask = cv2.filter2D(filtered_mask, -1, kernel)

    # Apply corrections on foreground_mask
    correction_mask = np.zeros(filtered_mask.shape, dtype = np.uint8)
    correction_mask[filtered_mask <= 255*5] = 255
    correction_mask = cv2.bitwise_or(correction_mask, correction_mask, mask = puzzle_mask)
    foreground_mask[correction_mask == 255] = 0

    # Apply corrections on puzzle_mask
    puzzle_mask[filtered_mask <= 255*5] = 0

  # Delete borders from foreground
  foreground_mask[puzzle_mask == 255] = 0  
  foreground_mask = np.dstack([foreground_mask]*3)
  return puzzle_image, puzzle_mask, foreground_mask

# Input: Image and padding shape
# Output: Padded image
def add_padding(image, padding_shape):

  try:
    assert (padding_shape[0] >= image.shape[0] and padding_shape[1] >= image.shape[1]), 'Padding size must be equal or bigger than image size.'
  except AssertionError as err:
    print('Error: {}'.format(err))
    exit(1)

  # Create empty image
  padded_image = np.zeros((padding_shape[0], padding_shape[1], image.shape[2]), dtype=np.uint8)

  # Calculate image left corner y position
  left_corner_y = padding_shape[0] - image.shape[0]
  left_corner_y //= 2

  # Calculate image left corner x position
  left_corner_x = padding_shape[1] - image.shape[1]
  left_corner_x //= 2

  # Center image on padded image
  padded_image[left_corner_y:left_corner_y + image.shape[0], left_corner_x:left_corner_x + image.shape[1]] = image

  return padded_image, left_corner_y, left_corner_x

def index_to_coordinate(index, piece_size):
  i, j = index
  i_coord = i * piece_size + piece_size // 2
  j_coord = j * piece_size + piece_size // 2
  return i_coord, j_coord 

# Function that create translation offsets for removed pieces
# Returns a tuple (translate_row, translate_col) representing the translation on both axis
def make_transformations(piece_size, padding_height, padding_width, image_height, image_width):

  offsets = []
  ind = 0

  # Create all transformations on left and right sides
  while ind*2*piece_size <= image_height:

    width_offset = 0
    while width_offset < padding_width - piece_size:

      # Left
      offsets.append((np.random.randint(ind*2*piece_size, ind*2*piece_size+piece_size//2), - width_offset - np.random.randint(piece_size, 2*piece_size - piece_size//2)))

      # Right
      offsets.append((np.random.randint(ind*2*piece_size, ind*2*piece_size+piece_size//2), image_width + width_offset + np.random.randint(piece_size, 2*piece_size - piece_size//2)))

      width_offset += 2*piece_size

    ind += 1
  
  ind = 0

  # Create all transformations for up and down
  while ind*2*piece_size < image_width:

    height_offset = 0
    while height_offset < padding_height - piece_size:
      # Up
      offsets.append((- height_offset - np.random.randint(piece_size, 2*piece_size - piece_size//2), np.random.randint(ind*2*piece_size, ind*2*piece_size+piece_size//2)))

      # Down
      offsets.append((height_offset + image_height + np.random.randint(piece_size, 2*piece_size - piece_size//2), np.random.randint(ind*2*piece_size, ind*2*piece_size+piece_size//2)))

      height_offset += 2*piece_size

    ind += 1
  
  return np.array(offsets)

# Rotate a point around the center by the given value of degrees
def rotate_around(center, point, degree):

  y,x = point
  cy, cx = center
  radians = np.pi * (degree/180)
  
  # Translate point to space where center is the origin
  y -= cy
  x -= cx

  # Rotate
  y, x = y*np.cos(radians) - x*np.sin(radians), y*np.sin(radians) + x*np.cos(radians)

  # Translate back to original space
  y = int(y) + cy
  x = int(x) + cx

  return y,x

def piece_selection(indices, image, piece_size, puzzle_j, puzzle_i, n_moving_pieces):
  board = image.copy()
  number_of_pieces = len(indices[:,0])
  # Set the text format
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_size = piece_size / 100
  font_color = (0,0,0)
  
  # Add a number for each piece, according to the indices' rows
  for ind in range(0, number_of_pieces):
    k,l = index_to_coordinate(indices[ind], piece_size)
    n_piece = str(ind)
    textsize = cv2.getTextSize(n_piece, font, font_size, 2)[0]
    k += (piece_size - textsize[0]) // 2 + puzzle_j - piece_size //2
    l += (piece_size + textsize[1]) // 2 + puzzle_i - piece_size //2
    
    # Add a background to the text, for an easier visualization
    box_coords = ((k, l + 1), (k + textsize[0] - 2,l - textsize[1] - 2))
    cv2.rectangle(board, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)
    cv2.putText(board, n_piece, (k,l), font, font_size, font_color, 1, cv2.LINE_AA)
  
  order = []
  while True:
    
    # Show the board with the numbers
    cv2.imshow('Removed Pieces Selection', board)
    cv2.waitKey(1000)

    # Get user input
    try:
      order = list(map(int,input('\nEnter the {0} numbers separated by spaces or enter \'-1\' to exit : '.format(n_moving_pieces)).strip().split()))
      
      # If user digits -1, exit program
      if(any(x == -1 for x in order)):
        exit(0)

      # If any number is greater or equal than the number of pieces, or less than zero, ignore input and try again
      elif(any(x < -1 for x in order) or any(x >= number_of_pieces for x in order)):
        print('All numbers must be between 0 and {0}'.format(number_of_pieces-1))

      # If the user did not give the correct amount of numbers, ignore and try again
      elif(len(order) != n_moving_pieces): 
        print('Wrong number of pieces. You must type {0} different numbers.'.format(n_moving_pieces))

      # If the user gives duplicate numbers, ignore and try again
      elif len(order) != len(set(order)):
        print('The list contains duplicates. You must type {0} different numbers.'.format(n_moving_pieces))
      
      # If list is ok, break the loop
      elif((len(order) == n_moving_pieces) and (all(x < number_of_pieces for x in order)) and all(x >= 0 for x in order)):
        break

    except ValueError:
      print('Not valid numbers.')
      continue
  
  cv2.destroyAllWindows()
  indices[:,0], indices[:,1] = indices[:,1], indices[:,0].copy()

  # Return the values
  return np.array(order)