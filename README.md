# sudoku-solver
Solves sudoku in a real time webcam feed

sudoku_solver.py takes a real time webcam feed and extracts sudoku from the frames and prints the solution on the screen

The algorithm used to solve the sudoku is is by Peter Norvig and solves sudoku problems in under 0.01 seconds

To train the digit recognizer, I have curated a data using the Char74K Data repository by and manually segmenting a very small subset from it that match the normal font digits in a sudoku.
The training has been done using a custom architecture inspired from the seminal LeNet Architecture (that was origninally used by Yann LeCun et al. to solve the MNIST problem)

I would like to thank Dr. Adrian Rosebrock, Dr. Satya Mallick, and Ashwin Pajankar's youtube channel that gave me all the knowledge for Image Processing using opencv and deep learning!

Would like to cite Dr. Adrian Rosebrock's DL4CV book, which motivated me to take up such personal projects.

@book{rosebrock_dl4cv,ho
  author={Rosebrock, Adrian},
  title={Deep Learning for Computer Vision with Python},
  year={2019},
  edition={3.0.0},
  publisher={PyImageSearch.com}
}
