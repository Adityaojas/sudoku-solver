# SUDOKU SOLVER
# BY - ADITYAOJAS SHARMA

# import the libraries
from tensorflow.keras.models import load_model
from imutils.perspective import four_point_transform
from norvig import solve
import norvig
import numpy as np
import imutils
import cv2

# load the model
model = load_model('model/best_model_lenet_aug_loss_3.model')

# webcam feed
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

# iterate through the frames
while ret:
    
    # Read the frame and preprocess
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 1000)
    (height, width) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Adaptive threshold for to obtain a Binary Image from the frame
    # followed by Hough lines probablistic transform to highlight the straight line proposals
    
    adTh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 5)
    lines = cv2.HoughLinesP(adTh,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    img_lines = adTh.copy()
    
    # there might be some cases where the lines are not found
    # in the frame. To avoid an error we'll use try & except
    try:
        for x1, y1, x2, y2 in lines[:,0,:]: 
            cv2.line(img_lines,(x1,y1), (x2,y2), (255,255,255),2)
    except:
        pass
    
    # Find the proposals sudoku contour
    cnts = cv2.findContours(img_lines.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    # Define the important copies of images and arrays for final output
    cn_img = frame.copy()
    final_im = frame    
    grid_digits = ['0']*81
    rois = []
    
    # Iterate through the biggest few contours and find the one for sudoku
    # We assume that the biggest square in the image is our sudoku
    for (i,c) in enumerate(cnts):
        peri = cv2.arcLength(c, True)    
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # A square contour must have:
        # 1) 4 corners &
        # 2) aspect ratio ~ 1
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            if ar >= 1.2 or ar <= 0.8:
                continue
            
            # If aspect ratio suits, continue
            mainCnt = approx
            full_coords = mainCnt.reshape(4, 2)
            
            # Draw the contours in such a way that our image is blanks where the sudoku is found
            cv2.drawContours(cn_img, [mainCnt], -1, (0, 0, 0), -1)
            
            
            # 4 point transform to obtain a top-down image for our found sudoku
            sudoku = four_point_transform(img_lines, mainCnt.reshape(4, 2))
            sudoku_clr = four_point_transform(frame, mainCnt.reshape(4, 2))
            sud_c = sudoku.copy()
            
            # Highlight the grid for sudoku
            # First, find the horizontal and Vertical edges
            horizontal = np.copy(sud_c)
            vertical = np.copy(sud_c)
            
            cols = horizontal.shape[1]
            horizontal_size = cols // 10
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)
            
            rows = vertical.shape[0]
            verticalsize = rows // 10
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)
            
            # Then, add the horizontal and vertical edge image using bitwise_or
            grid = cv2.bitwise_or(horizontal, vertical)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            grid = cv2.dilate(grid, kernel)
            
            # The grid we obtain might be a bit thicker than the original grid
            # so remove the unwanted thickness using bitwise_and
            grid = cv2.bitwise_and(grid, sudoku)
            
            # Finally, subtract the grid from our sudoku image and obtain
            # an image with just numbers            
            num = cv2.bitwise_xor(sud_c, grid)
            
            # Obtain the corners of our top-down sudoku with respect to 
            # the order of the coordinates obtained during the perspective transform
            if (full_coords[0][0])**2+(full_coords[0][1])**2 < (full_coords[1][0])**2+(full_coords[1][1])**2:
                sud_coords = np.array([[0, 0], [0, num.shape[0]], [num.shape[1], num.shape[0]], [num.shape[1], 0]])
            else:
                sud_coords = np.array([[num.shape[1], 0], [0, 0], [0, num.shape[0]], [num.shape[1], num.shape[0]]])
            """
            num_2 = cv2.cvtColor(num, cv2.COLOR_GRAY2BGR)
            """
            
            # Obtain the shape of our grid-less proposal
            num_r = num.shape[0]
            num_c = num.shape[1]
            num_side = min(num_r, num_c)
            
            # We'll be sliding a window through our proposal,
            # so that we obtain 81 sub squares. 
            windowsize_r = (num_r // 9) - 1
            windowsize_c = (num_c // 9) - 1
            
            window_area = windowsize_r * windowsize_c
            
            # Define a smallest proposal area as a threshold area for the digit contour
            smallest_prop_area = window_area // 16
            
            # In case our grid isn't eliminated completely,
            # to avoid interference, we define a buffer to be subtracted from the window sides
            buffer_r = windowsize_r // 9 
            buffer_c = windowsize_c // 9
            
            # Define a counter, i, to keep a check on the number of windows
            i=-1                                    
            
            # Start iterating!
            for r in range(0,num.shape[0] - windowsize_r, windowsize_r):
                for c in range(0,num.shape[1] - windowsize_c, windowsize_c):
                    
                    # Keep a list of all the windows in a list
                    rois.append([r, r+windowsize_r, c, c+windowsize_c])
                    
                    i+=1
                    
                    # Define our window
                    window = num[r+buffer_r:r-buffer_r+windowsize_r, c+buffer_c:c-buffer_c+windowsize_c]                    
                    
                    # Find our contour proposals in each window
                    proposals = cv2.findContours(window, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    proposals = imutils.grab_contours(proposals)
                    
                    # Iterate through these proposals & check if the bounding rectangle
                    # has an area greater than our threshold area
                    if len(proposals) > 0:
                        digit = sorted(proposals, key = cv2.contourArea, reverse = True)[0]
                        perimeter = cv2.arcLength(digit, True)
                        approx_shape = cv2.approxPolyDP(digit, 0.02 * perimeter, True)
                        bound_rect = cv2.boundingRect(approx_shape)
                        
                        rect_area = bound_rect[2] * bound_rect[3]
                    
                        if rect_area < smallest_prop_area:
                            continue
                        
                        
                        (x,y,w,h) = bound_rect
                        
                        # Define a single side to avoid errors if the bounding rectangle coordinates
                        # lie outside the image
                        s = 2 * (max(w,h) // 2)
                        
                        
                        cv2.rectangle(sudoku_clr, (c+x+buffer_c, r+y+buffer_r),
                                      (c+x+w+buffer_c, r+y+h+buffer_r),
                                      (0, 255, 0), 1)
                        
                        """
                        cv2.rectangle(num_2, (c+x+buffer_c, r+y+buffer_r),
                                     c+x+w+buffer_c, r+y+h+buffer_r),
                                     (0, 255, 0), 1)
                        """
                        
                        # Transform the bounding rectangle coordinates
                        # to represent square structure
                        r_start = r+y+(h//2)-(s//2)-(2*buffer_r)
                        if r_start < 0:
                            r_start = 0
                        
                        r_end = r+y+(h//2)+(s//2)+(3*buffer_r)
                        if r_end > num_r:
                            r_end = num_r
                        
                        
                        c_start = c+x+(w//2)-(s//2)-(2*buffer_c)
                        if c_start < 0:
                            c_start = 0
                            
                        c_end = c+x+(w//2)+(s//2)+(3*buffer_c)
                        if c_end > num_c:
                            c_end = num_c
                        
                        # Define the proposal area
                        prop = num[r_start:r_end, c_start:c_end]
                        
                        # Sometimes the proposal area might be left empty due to various
                        # unavoidable reasons, like brightness, illumination, etc.
                        # To avoid errors while prediction, we'll use try & except
                        try:
                            prop = cv2.resize(prop, (28, 28), cv2.INTER_AREA)
                            prop = np.atleast_3d(prop)
                            prop = np.expand_dims(prop, axis = 0)   
                                                             
                            pred = model.predict(prop).argmax(axis=1)
                            """
                            print(pred)
                            """
                            
                            try:
                                grid_digits[i] = (str(int(pred[0])+1))
                            except:
                                pass
                        
                        except:
                            pass
            
            # Finally, we have the grid.
            # Predict the solution using the Norvig Sudoku Solving Algorithm
            if len(grid_digits) == 81:
                solved = solve(grid_digits)
                
                if solved != False:
                    solved = list(solved.values())
                    
                    # Print the text on our image
                    for e in range(81):
                        if grid_digits[e]!='0':
                            continue
                        sudoku_clr=cv2.putText(sudoku_clr, solved[e], ((rois[e][2]+rois[e][3])//2, rois[e][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2)
                        
                        # Now define the homography map and apply warp perspective
                        # to fit the top-down sudoku back on our frame.
                        h, mask = cv2.findHomography(sud_coords, full_coords)
                        im_out = cv2.warpPerspective(sudoku_clr, h, (width, height))
                        
                        final_im = im_out + cn_img
                        
            break
    
    """           
    print(grid_digits)
    cv2.imshow("Contoured Image", cn_img)
    cv2.imshow("sudoku_binary", im_out)    
    cv2.imshow("cn_num", cn_num)
    """
    cv2.imshow("sudoku", final_im)
    
    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()


