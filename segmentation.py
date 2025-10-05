import os
import numpy as np
import torch
import matplotlib.pyplot as plt 
from PIL import Image
import random
import cv2 as cv

from sam2.build_sam import build_sam2 
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# Default choice of checkpoint and config
sam2_checkpoint = os.path.expanduser("~/Desktop/MasterThesis/data/sam_ckpts/sam2.1_hiera_large.pt")
model_cfg = "sam2.1_hiera_l.yaml"


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def automatic_masks(img: np.ndarray, checkpoint: str = sam2_checkpoint, config: str = model_cfg):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")

    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3)

    sam2 = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks = mask_generator.generate(img)

    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def mouse_callback_left(event, x, y, flags, param):
    points, img_display = param
    if event == cv.EVENT_LBUTTONDOWN:
        points.append([x,y])
        cv.circle(img_display, (x, y), 5, (0, 255, 0), -1)

# def gen_segmentation():



if __name__ == "__main__":

    dataset_path = os.path.expanduser("~/Desktop/MasterThesis/data/datasets/Robofarmer-II")
    train_imgs_path = os.path.join(dataset_path, "inactive_images/train_images")

    imgs = os.listdir(train_imgs_path)
    img = Image.open(os.path.join(train_imgs_path, random.choice(imgs)))
    img = np.array(img.convert("RGB"))
    img_display = cv.resize(cv.cvtColor(img.copy(), cv.COLOR_RGB2BGR), (900,700), interpolation=cv.INTER_CUBIC)
    img = cv.resize(img.copy(), (900,700), interpolation=cv.INTER_CUBIC)
    points = []
    cv.namedWindow("Segmentation point choice", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Segmentation point choice", mouse_callback_left, (points, img_display))
    
    while (True):
        cv.imshow("Segmentation point choice", img_display)
        key = cv.waitKey(1) & 0xFF 
        if key == ord("q"):
            break
    
    cv.destroyAllWindows()

    input_points = np.array(points)
    # Trying multi-label first
    input_labels = np.arange(len(points))

    

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")

    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 0:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3)

    # input_points = np.flip(input_points, axis=1)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    show_masks(img, masks, scores, point_coords=input_points, input_labels=input_labels, borders=True)

    # automatic_masks(img)

