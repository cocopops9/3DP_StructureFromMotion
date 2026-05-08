from pathlib import Path
import argparse

import cv2
import torch

from models.matching import Matching

#fix params

#dataset dir
DATASET_DIR = Path("../3DP_lab_2/datasets")

#same focal scale of the matcher
FOCAL_SCALE = 1.1
MAX_KEYPOINTS = 10000 #as in ORB

#SuperPoint/SuperGlue params (default value)
NMS_RADIUS = 2
KEYPOINT_THRESHOLD = 0.0001
MATCH_THRESHOLD = 0.25
SUPERGLUE_WEIGHTS = "indoor" #we use pretrained weights in indoor modality

#vm runtime
DEVICE = "cpu" #VM is not gpu compatible

#helper functions with same logic of c++ counterpart
#io_utils
def loadCameraParams(file_name):
    fs = cv2.FileStorage(str(file_name), cv2.FILE_STORAGE_READ)

    width = int(fs.getNode("width").real())
    height = int(fs.getNode("height").real())

    intrinsics_matrix = fs.getNode("K").mat()
    dist_coeffs = fs.getNode("D").mat()

    fs.release()

    image_size = (width, height)
    return image_size, intrinsics_matrix, dist_coeffs

def readFileNamesFromFolder(input_folder_name):
    p = Path(input_folder_name)

    names = [str(entry) for entry in p.iterdir()]
    names.sort()

    return names

def buildNewIntrinsicsMatrix(intrinsics_matrix):

    new_intrinsics_matrix = intrinsics_matrix.copy()
    new_intrinsics_matrix[0, 0] *= FOCAL_SCALE
    new_intrinsics_matrix[1, 1] *= FOCAL_SCALE

    return new_intrinsics_matrix

def readUndistortedImage(filename, intrinsics_matrix, dist_coeffs, new_intrinsics_matrix):
    img = cv2.imread(filename)

    und_img = cv2.undistort(img, intrinsics_matrix, dist_coeffs, None, new_intrinsics_matrix)
    return und_img

#specific functions for superpoint/superglue

#Superpoint works on normalized grayscae imgs in the form of tensors
def imageToTensor(image_bgr):
    
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    tensor = torch.from_numpy(gray_img/255.0).float()
    tensor = tensor[None, None] ##add dimension to tensor

    return tensor   #in the form [batch size, channels, height, width]

#save SuperPoint keypoints in a .txt file
def saveFeatures(output_file, keypoints, image_bgr):
    h, w = image_bgr.shape[:2]

    with open(output_file, "w") as f:
        f.write(str(keypoints.shape[0]) + "\n")

        for k in range(keypoints.shape[0]):
            x = float(keypoints[k, 0])
            y = float(keypoints[k, 1])

            #to retrieve color i need discretization
            xi = int(round(x))
            yi = int(round(y))

            xi = max(0, min(w - 1, xi)) #clamping to avoid exiting img border
            yi = max(0, min(h - 1, yi))
            
            b, g, r = image_bgr[yi, xi]
            f.write(f"{x} {y} {int(b)} {int(g)} {int(r)}\n")

#save SuperGlue matches in a .txt file
def saveMatches(output_file, matches0, scores0):
    valid_indices = []

    for idx0 in range(matches0.shape[0]):
        idx1 = int(matches0[idx0]) #find correspective keypoint in img 1 given the one on img 0

        if idx1 >= 0: #if the match exist
            valid_indices.append(idx0)
    
    with open(output_file, "w") as f:
        f.write(str(len(valid_indices)) + "\n")

        for idx0 in valid_indices:
            idx1 = int(matches0[idx0])
            score = float(scores0[idx0])

            f.write(f"{idx0} {idx1} {score}\n")

#main
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_folder", required=True, help="Name of the img directory: images_1, images_2...")

    parser.add_argument("--calib", required=True, help="yaml path file")

    args = parser.parse_args()

    #dataset path
    images_dir = DATASET_DIR/args.images_folder

    #calibration file path
    calib_path = Path(args.calib)

    #output files directory
    output_root = DATASET_DIR/("superglue_" + args.images_folder)
    features_dir = output_root/"features"
    matches_dir = output_root/"matches"

    features_dir.mkdir(parents=True, exist_ok=True)
    matches_dir.mkdir(parents=True, exist_ok=True)

    #read calibration and img list
    image_size, intrinsics_matrix, dist_coeffs = loadCameraParams(calib_path)
    images_names = readFileNamesFromFolder(images_dir)
    new_intrinsics_matrix = buildNewIntrinsicsMatrix(intrinsics_matrix)

    #create superpoint and superglue model
    config = {
        "superpoint": {
            "nms_radius": NMS_RADIUS,
            "keypoint_threshold": KEYPOINT_THRESHOLD,
            "max_keypoints": MAX_KEYPOINTS,
        },
        "superglue": {
            "weights": SUPERGLUE_WEIGHTS,
            "sinkhorn_iterations": 20,
            "match_threshold": MATCH_THRESHOLD,
        },
    }

    matching = Matching(config).eval().to(DEVICE)

    image_tensors = []
    features = []
    
    #features extraction
    print("\nExtracting SuperPoint features")
    for i, image_name in enumerate(images_names):
        print("Image", i, ":", image_name)

        img = readUndistortedImage(image_name, intrinsics_matrix, dist_coeffs, new_intrinsics_matrix)

        tensor = imageToTensor(img).to(DEVICE)

        #inference
        with torch.no_grad():
            pred = matching.superpoint({"image": tensor})

        keypoints = pred["keypoints"][0]
        scores = pred["scores"][0]
        descriptors = pred["descriptors"][0]

        #saving keypoints
        file_without_extension = Path(image_name).stem
        feature_file = features_dir/(file_without_extension + ".txt")

        saveFeatures(feature_file, keypoints.numpy(), img)

        #keep tensor and features in memory for superglue phase
        image_tensors.append(tensor)
        features.append({
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        })

        print("  keypoints:", keypoints.shape[0])
        print("  saved:", feature_file)

    #feature matching
    print("\nMatching all image pairs with SuperGlue")

    for i in range(len(images_names)-1):
        for j in range(i+1, len(images_names)):
            img_i = Path(images_names[i]).stem
            img_j = Path(images_names[j]).stem

            print("Matching", img_i, "with", img_j)

            #Superglue input
            data = {
                "image0": image_tensors[i],
                "image1": image_tensors[j],

                "keypoints0": [features[i]["keypoints"]],
                "keypoints1": [features[j]["keypoints"]],

                "scores0": [features[i]["scores"]],
                "scores1": [features[j]["scores"]],

                "descriptors0": [features[i]["descriptors"]],
                "descriptors1": [features[j]["descriptors"]],
            }

            #superglue inference
            with torch.no_grad():
                pred = matching(data)

            matches0 = pred["matches0"][0]
            scores0 = pred["matching_scores0"][0]

            matches_file = matches_dir/(img_i + "_" + img_j + ".txt")

            saveMatches(matches_file, matches0.numpy(), scores0.numpy())

            valid_matches = 0
            for k in range(len(matches0)):
                if matches0[k] > -1:
                    valid_matches = valid_matches + 1

            print("valid matches:", valid_matches)
            print("saved:", matches_file)
    print("\nDone!")

if __name__ == "__main__":
    main()







    

