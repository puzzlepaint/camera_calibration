# Copyright 2019 ETH Zürich, Thomas Schöps
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
import math
import sys

import numpy as np

# This requires Matplotlib
from matplotlib.pyplot import imread

# This requires reportlab, installed like this:
# sudo pip3 install reportlab
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch, cm, mm

def GetStarCoord(square_length, i, num_star_segments, center_x, center_y):
  angle = (2 * math.pi) * i / num_star_segments
  x = math.sin(angle)
  y = math.cos(angle)
  
  max_abs_x = max(abs(x), abs(y))
  x /= max_abs_x
  y /= max_abs_x
  
  return (center_x - 0.5 * square_length * x,
          center_y + 0.5 * square_length * y)


if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description="Create calibration patterns.")
  parser.add_argument("--tag36h11_path", required=True,
                      help="Path to a folder containing the 36h11 AprilTag images. May be downloaded from: https://github.com/AprilRobotics/apriltag-imgs")
  parser.add_argument("--output_base_path", required=True,
                      help="Base path to the PDF and YAML output files (excluding the file extensions).")
  parser.add_argument("--paper_size", default="A4",
                      help="Paper size; supported values: A4, letter, or <width>x<height> with the dimensions in centimeters, for example: 20.5x40")
  parser.add_argument("--num_star_segments", default="16",
                      help="Number of segments of each star in the pattern. Refers to the sum of black and white segments. 4 would give a checkerboard.")
  parser.add_argument("--apriltag_index", default="0",
                      help="Index of the AprilTag to use for the pattern.")
  parser.add_argument("--margin_in_cm", default="0.4",
                      help="Page margin in centimeters.")
  parser.add_argument("--approx_square_length_in_cm", default="1.2",
                      help="Approximate star square length in centimeters. May get slightly modified such that the squares exactly fit into the print area.")
  parser.add_argument("--apriltag_length_in_squares", default="4",
                      help="Length of the AprilTag measured in star squares.")
  
  # Parse and check arguments
  args = parser.parse_args()
  
  num_star_segments = int(args.num_star_segments)
  apriltag_index = int(args.apriltag_index)
  margin_in_cm = float(args.margin_in_cm)
  approx_square_length_in_cm = float(args.approx_square_length_in_cm)
  apriltag_length_in_squares = int(args.apriltag_length_in_squares)
  
  pagesize = A4
  pagesize_r = (pagesize[0]*5,pagesize[1]*5)
  if args.paper_size == "A4":
    pagesize = A4
  elif args.paper_size == "letter":
    pagesize = letter
  elif 'x' in args.paper_size:
    width, height = args.paper_size.split('x')
    width = float(width) * cm
    height = float(height) * cm
    pagesize = (width, height)
  else:
    print("Error: The given paper size (" + args.paper_size + ") must be either A4 or letter.")
    sys.exit(1)
  pagesize = pagesize_r

  pdf_path = args.output_base_path + 'out.pdf'
  metadata_path = args.output_base_path + 'out.yaml'
  
  tag_path = os.path.join(args.tag36h11_path, 'tag36_11_{:0>5d}.png'.format(apriltag_index))
  
  if num_star_segments < 4:
    print('Error: The number of star segments must be larger or equal to four.')
    sys.exit(1)
  
  if num_star_segments % 4 != 0:
    print('Warning: The number of star segments must be divisible by four for the symmetry-based detector.')
  
  if not os.path.exists(tag_path):
    print('Error: Required file does not exist: ' + tag_path)
    sys.exit(1)
  
  # Set up page. (0, 0) is at the bottom-left of the page.
  c = canvas.Canvas(pdf_path, pagesize=pagesize)
  c.setFillColorRGB(0, 0, 0)
  
  width, height = pagesize
  margin = margin_in_cm * cm
  
  start_x = margin
  end_x = width - margin
  
  start_y = height - margin
  end_y = margin
  
  print_area_width = abs(end_x - start_x)
  print_area_height = abs(end_y - start_y)
  
  # Determine the checkerboard resolution
  approx_square_length = approx_square_length_in_cm * cm
  squares_length_1 = print_area_width / round(print_area_width / approx_square_length)
  squares_length_2 = print_area_height / round(print_area_height / approx_square_length)
  
  square_length = min(squares_length_1, squares_length_2)
  squares_x = math.floor(print_area_width / square_length)
  squares_y = math.floor(print_area_height / square_length)
  
  unused_x = print_area_width - squares_x * square_length
  pattern_start_x = start_x + 0.5 * unused_x
  
  unused_y = print_area_height - squares_y * square_length
  pattern_start_y = start_y - 0.5 * unused_y
  
  # Draw AprilTag in the middle
  clip_path = c.beginPath()
  
  im = imread(tag_path).astype(np.uint8)
  tag_width = im.shape[0]
  tag_height = im.shape[1]
  if tag_width != tag_height:
    print('Non-square tags are not supported')
    sys.exit(1)
  
  tag_x = squares_x // 2 - apriltag_length_in_squares // 2
  tag_start_x = pattern_start_x + tag_x * square_length
  tag_y = squares_y // 2 - apriltag_length_in_squares // 2
  tag_start_y = pattern_start_y - tag_y * square_length
  
  tag_square_length = apriltag_length_in_squares * square_length / tag_width
  for x in range(0, tag_width):
    for y in range(0, tag_height):
      if im[y][x][0] == 0:
        c.rect(tag_start_x + x * tag_square_length,
                tag_start_y - y * tag_square_length - tag_square_length,
                tag_square_length,
                tag_square_length,
                stroke=0,
                fill=1)
  
  clip_path.moveTo(tag_start_x, tag_start_y)
  clip_path.lineTo(tag_start_x + tag_width * tag_square_length, tag_start_y)
  clip_path.lineTo(tag_start_x + tag_width * tag_square_length, tag_start_y - tag_height * tag_square_length)
  clip_path.lineTo(tag_start_x, tag_start_y - tag_height * tag_square_length)
  clip_path.lineTo(tag_start_x, tag_start_y)
  
  pattern_end_x = end_x - 0.5 * unused_x
  pattern_end_y = end_y + 0.5 * unused_y
  clip_path.moveTo(pattern_start_x, pattern_start_y)
  clip_path.lineTo(pattern_end_x, pattern_start_y)
  clip_path.lineTo(pattern_end_x, pattern_end_y)
  clip_path.lineTo(pattern_start_x, pattern_end_y)
  clip_path.lineTo(pattern_start_x, pattern_start_y)
  
  # Draw checkerboard
  c.clipPath(clip_path, stroke=0, fill=0)
  
  for x in range(-1, squares_x):
    for y in range(0, squares_y + 1):
      center_x = pattern_start_x + (x + 1) * square_length
      center_y = pattern_start_y - y * square_length
      
      path = c.beginPath()
      
      # Draw all black segments
      for segment in range(0, num_star_segments, 2):
        path.moveTo(center_x, center_y)
        
        sc1 = GetStarCoord(square_length, segment, num_star_segments, center_x, center_y)
        path.lineTo(sc1[0], sc1[1])
        
        # Add point at the square corner?
        angle1 = (2 * math.pi) * (segment) / num_star_segments
        angle2 = (2 * math.pi) * (segment + 1) / num_star_segments
        if math.floor((angle1 - math.pi / 4) / (math.pi / 2)) != math.floor((angle2 - math.pi / 4) / (math.pi / 2)):
          corner_angle = (math.pi / 4) + (math.pi / 2) * math.floor((angle2 - math.pi / 4) / (math.pi / 2))
          corner_x = math.sin(corner_angle)
          corner_y = math.cos(corner_angle)
          normalizer = abs(corner_x)
          corner_x /= normalizer
          corner_y /= normalizer
          corner_coord = (center_x - 0.5 * square_length * corner_x,
                          center_y + 0.5 * square_length * corner_y)
          path.lineTo(corner_coord[0], corner_coord[1])
        
        sc2 = GetStarCoord(square_length, segment + 1, num_star_segments, center_x, center_y)
        path.lineTo(sc2[0], sc2[1])
        
        path.lineTo(center_x, center_y)
      
      c.drawPath(path, stroke=0, fill=1)
  
  # Write metadata
  with open(metadata_path, 'wb') as metadata_file:
    metadata_file.write(bytes('num_star_segments: ' + str(num_star_segments) + '\n', 'UTF-8'))
    metadata_file.write(bytes('squares_x: ' + str(squares_x) + '\n', 'UTF-8'))
    metadata_file.write(bytes('squares_y: ' + str(squares_y) + '\n', 'UTF-8'))
    metadata_file.write(bytes('square_length_in_meters: ' + str(0.01 * square_length / cm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('page:\n', 'UTF-8'))
    metadata_file.write(bytes('    width_mm: ' + str(width / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    height_mm: ' + str(height / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    pattern_start_x_mm: ' + str(pattern_start_x / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    pattern_start_y_mm: ' + str((height - pattern_start_y) / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    pattern_end_x_mm: ' + str(pattern_end_x / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    pattern_end_y_mm: ' + str((height - pattern_end_y) / mm) + '\n', 'UTF-8'))
    metadata_file.write(bytes('apriltags:\n', 'UTF-8'))
    metadata_file.write(bytes('  - tag_x: ' + str(tag_x) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    tag_y: ' + str(tag_y) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    width: ' + str(apriltag_length_in_squares) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    height: ' + str(apriltag_length_in_squares) + '\n', 'UTF-8'))
    metadata_file.write(bytes('    index: ' + str(apriltag_index) + '\n', 'UTF-8'))
  
  # Save the page
  c.setTitle('Calibration pattern #' + str(apriltag_index))
  c.setAuthor('Calibration pattern generation script')
  
  c.showPage()
  c.save()
  
  print('Successfully generated pattern:\n' + pdf_path + '\nwith metadata:\n' + metadata_path)
