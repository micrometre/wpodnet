# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long
from os.path import splitext
import cv2
import numpy as np
from keras.models import model_from_json
import keras


class Label:
    """Class to represent a bounding box label with class, coordinates, and probability"""
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl  # top-left coordinates [x,y]
        self.__br = br  # bottom-right coordinates [x,y]
        self.__cl = cl  # class
        self.__prob = prob  # probability/confidence score

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
        self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl  # width and height of the box

    def cc(self): return self.__tl + self.wh() / 2  # center coordinates

    def tl(self): return self.__tl  # get top-left coordinates

    def br(self): return self.__br  # get bottom-right coordinates

    def tr(self): return np.array([self.__br[0], self.__tl[1]])  # top-right coordinates

    def bl(self): return np.array([self.__tl[0], self.__br[1]])  # bottom-left coordinates

    def cl(self): return self.__cl  # get class

    def area(self): return np.prod(self.wh())  # area of the box

    def prob(self): return self.__prob  # get probability

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    """Extended Label class that uses points instead of just top-left and bottom-right coordinates"""
    def __init__(self, cl, pts, prob):
        self.pts = pts  # four corner points of the bounding box
        tl = np.amin(pts, axis=1)  # calculate top-left from points
        br = np.amax(pts, axis=1)  # calculate bottom-right from points
        Label.__init__(self, cl, tl, br, prob)


def im2single(Image):
    """Convert image to single precision floating point format (0-1 range)"""
    return Image.astype('float32') / 255


def getWH(shape):
    """Get width and height from image shape (returns as numpy array)"""
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    """Calculate Intersection Over Union (IOU) between two bounding boxes"""
    wh1, wh2 = br1-tl1, br2-tl2
    assert((wh1 >= 0).all() and (wh2 >= 0).all())
    
    # Calculate intersection area
    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    
    # Calculate union area
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    
    return intersection_area/union_area


def IOU_labels(l1, l2):
    """Calculate IOU for two Label objects"""
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    """Non-Maximum Suppression to filter overlapping bounding boxes"""
    SelectedLabels = []
    # Sort labels by probability in descending order
    Labels.sort(key=lambda l: l.prob(), reverse=True)
    
    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def load_model(json_path, weights_path):
    """Load WPOD-NET model from JSON and weights files"""
    with open(json_path) as json_file:
        model_json = json_file.read()
    
    # Create model from JSON and load weights
    model = model_from_json(model_json, custom_objects={'Model': keras.models.Model})
    model.load_weights(weights_path)
    return model


def find_T_matrix(pts, t_pts):
    """Find transformation matrix between two sets of points"""
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T
        
        # Build matrix for homography calculation
        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    # Solve using SVD
    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))  # Homography matrix
    return H


def getRectPts(tlx, tly, brx, bry):
    """Get rectangle points in homogeneous coordinates"""
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    """Normalize points based on side length and min/max values"""
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop


def reconstruct(I, Iresized, Yr, lp_threshold):
    """Reconstruct license plate from network output"""
    net_stride = 2**4  # Network stride (16)
    side = ((208 + 40)/2)/net_stride  # Side length calculation

    # Standard sizes for license plates
    one_line = (470, 110)  # Single line license plate size
    two_lines = (280, 200)  # Two-line license plate size

    Probs = Yr[..., 0]  # Probability scores from network
    Affines = Yr[..., 2:]  # Affine transformations from network

    # Find points where probability exceeds threshold
    xx, yy = np.where(Probs > lp_threshold)
    WH = getWH(Iresized.shape)
    MN = WH/net_stride

    vxx = vyy = 0.5  # Default values
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # Process affine transformation
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        pts_frontal = np.array(B*base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    # Apply NMS to filter overlapping detections
    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    print(final_labels_frontal)

    # Determine if license plate is single or double line
    out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    TLp = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            # Create target points for perspective transform
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)

            # Warp perspective to get license plate image
            Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
            TLp.append(Ilp)
    print(final_labels)
    return final_labels, TLp, lp_type


def detect_lp(model, I, max_dim, lp_threshold):
    """Main function to detect license plates in an image"""
    # Resize image while maintaining aspect ratio
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img

    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()

    Iresized = cv2.resize(I, (w, h))

    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))  # Add batch dimension

    # Run prediction
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)  # Remove batch dimension

    print(Yr.shape)

    # Reconstruct license plates from network output
    L, TLp, lp_type = reconstruct(I, Iresized, Yr, lp_threshold)

    return L, TLp, lp_type