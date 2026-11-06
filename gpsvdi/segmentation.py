import cv2
import numpy as np

DEBUG = False


def crop(image, x, y, w, h, padding=0):
    """Rectangular crop with optional padding, clipped to the image bounds.

    Replaces the single function `insect_coverage` imported from
    `insect_detection_functions`, which is absent from the repository. At
    padding=0 (the only value used here) this is a plain array slice.
    """
    y0, y1 = max(0, y - padding), min(image.shape[0], y + h + padding)
    x0, x1 = max(0, x - padding), min(image.shape[1], x + w + padding)
    return image[y0:y1, x0:x1]



class InsectCoverage:
    def __init__(self, image_file_name, boxes):
        """
        gets an image and its bounding boxes

        :param image_file_name: path to file
        :param boxes: a list of bounding boxes in format XYWH
        """
        self.image = cv2.imread(image_file_name)
        self.boxes = boxes
        self.height, self.width, _ = self.image.shape
        self.coverage_mask = np.uint8(np.zeros((self.height, self.width)))
        self.masks = None
        self.image_means = {}
        self.image_stds = {}
        self.image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV_FULL)
        im = {'h': (cv2.split(self.image_hsv))[0], 's': (cv2.split(self.image_hsv))[1],
              'v': (cv2.split(self.image_hsv))[2]}
        for i in 'hsv':
            self.image_means[i] = np.mean(im[i])
            self.image_stds[i] = np.std(im[i])

    def crop2mask(self, x: int, y: int, width: int, height: int, padding: int, debug: bool = DEBUG) -> list[np.array]:
        """
        from bounding box create a crop to convert it to a mask
        where white pixels represent an insect area in the image

        :param x: x-axis bounding box
        :param y: y-axis bounding box
        :param width:  width
        :param height: bounding box height
        :param padding: bounding a box padding
        :param debug: flag to debug
        :return: mask (ndarray)
        """
        # create crop
        c = crop(image=self.image, x=x, y=y, w=width, h=height, padding=padding)
        # de-noising crop
        c = cv2.fastNlMeansDenoisingColored(c, None, templateWindowSize=7, searchWindowSize=21, h=3, hColor=5)
        c_hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV_FULL)
        # initiate bounds
        lower = {}
        upper = {}
        f_lower = {
            'h': 0.17,
            's': 3,
            'v': 3
        }
        f_upper = {
            'h': 0.4,
            's': 2,
            'v': 1
        }
        # setting bounds per channel
        for i in 'hsv':
            max_val = 255
            lower[i] = np.round(max(self.image_means[i] - self.image_stds[i] * f_lower[i], 0))
            upper[i] = np.round(min(self.image_means[i] + self.image_stds[i] * f_upper[i], max_val))
        if debug:
            for i in ['h', 's', 'v']:
                print(f'{i}_mean: {np.round(self.image_means[i])}, {i}_std: {np.round(self.image_stds[i])}')
        # set lower bound to filter out
        lowers = np.array([lower['h'], lower['s'], lower['v']])
        # set upper bound to filter out
        uppers = np.array([upper['h'], upper['s'], upper['v']])
        # initial mask based on in range of background values, and by bitwise not convert to 'not background'
        in_range_mask = cv2.bitwise_not(cv2.inRange(c_hsv, lowers, uppers))
        # blur-dilate-blur-erode to remove small artifacts and unify white cluster close to each other
        in_range_mask = cv2.GaussianBlur(in_range_mask, (5, 5), sigmaX=-1, sigmaY=-1)
        in_range_mask = cv2.dilate(in_range_mask, None, iterations=1)
        in_range_mask = cv2.GaussianBlur(in_range_mask, (3, 3), sigmaX=-1, sigmaY=-1)
        in_range_mask = cv2.erode(in_range_mask, None, iterations=2)
        in_range_mask = np.uint8(np.where(in_range_mask > 30, 255, 0))

        # fine-tuning: cases the white background around the trap boundaries appears in the bounding box
        white_bg = cv2.bitwise_not(cv2.inRange(c_hsv, lowerb=(0, 0, lower['v']),
                                               upperb=(359, round(lower['s'] * 0.7), 255)))
        white_bg = cv2.GaussianBlur(white_bg, (5, 5), sigmaX=-1, sigmaY=-1)
        white_bg = cv2.dilate(white_bg, None, iterations=1)
        white_bg = cv2.GaussianBlur(white_bg, (5, 5), sigmaX=-1, sigmaY=-1)
        white_bg = cv2.erode(white_bg, None, iterations=1)
        white_bg = np.uint8(np.where(white_bg > 30, 255, 0))

        # intersect both masks
        mask = cv2.bitwise_and(in_range_mask, white_bg)

        # to remove small artifacts remained, find edges, filter contours too small area and darken them
        edges = cv2.Canny(mask, 0, 255)
        edges = cv2.dilate(edges, None, iterations=2)
        edges = cv2.erode(edges, None, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_size_bound = 50
        to_remove = [c for c in contours if cv2.contourArea(c) <= min_size_bound]

        crop_mask = cv2.drawContours(mask, to_remove, -1, 0, thickness=cv2.FILLED)

        # if debug:
        #     contours = [c for c in contours if cv2.contourArea(c) > min_size_bound]
        #     final = cv2.drawContours(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), contours, -1, (0, 255, 0, 0.5))
        #     fig, axes = plt.subplots(nrows=1, ncols=3, sharex='row', sharey='col', figsize=(20, 8))
        #     fig.suptitle(
        #         f'$mean:$ ' + ', '.join([f'{i}: {np.round(self.image_means[i])}' for i in 'hsv']) +
        #         f'$std:$  ' + ', '.join([f'{i}: {np.round(self.image_stds[i])}' for i in 'hsv']) +
        #         '\nlower: ' + ", ".join([str(lower[i]) for i in "hsv"]) +
        #         '\nupper: ' + ", ".join([str(upper[i]) for i in "hsv"])
        #     )
        #     axes[0].imshow(edges, cmap='gray')
        #     axes[0].title.set_text('edges: {len(edges)}')
        #     axes[1].imshow(crop_mask, cmap='gray')
        #     axes[1].title.set_text('mask')
        #
        #     axes[2].imshow(final)
        #     axes[2].title.set_text(f'on original crop input, contours: {len(contours)}')
        #     fig.show()

        return crop_mask

    def make_masks(self):
        masks = []
        for box in self.boxes:
            x, y, width, height = box

            if x + width > self.width:
                width = self.width - x
            if x < 0:
                x = 0
            if y + height > self.height:
                height = self.height - y
            if y < 0:
                y = 0

            mask = np.uint8(np.zeros((height, width)))
            crop_mask = self.crop2mask(x=x, y=y, width=width, height=height, padding=0)
            self.coverage_mask[y:y + height, x:x + width] =\
                cv2.bitwise_or(self.coverage_mask[y:y + height, x:x + width], crop_mask)
            mask = cv2.bitwise_or(mask, crop_mask)
            masks.append(mask > 0)

        self.masks = masks

    def total_coverage(self):
        return np.sum(self.coverage_mask>0)/(self.height*self.width)


