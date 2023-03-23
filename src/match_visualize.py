import cv2
import numpy as np
from typing import List
from pathlib import Path

def extract_orb_features(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(query_descriptors, dictionary_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_descriptors, dictionary_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def visualize_matches(query_img_path: str, best_match_path: str, query_keypoints, dictionary_keypoints, matches):
    query_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)
    dictionary_img = cv2.imread(best_match_path, cv2.IMREAD_GRAYSCALE)

    result = cv2.drawMatches(query_img, query_keypoints, dictionary_img, dictionary_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow("Matching result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('../output/output.jpg', result)

def find_best_match(query_img_path: str, dictionary_img_paths: List[str], visualize: bool = False):
    query_keypoints, query_descriptors = extract_orb_features(query_img_path)

    min_distance = float('inf')
    best_match_index = -1
    best_matches = None
    best_dictionary_keypoints = None

    for i, dictionary_img_path in enumerate(dictionary_img_paths):
        dictionary_keypoints, dictionary_descriptors = extract_orb_features(dictionary_img_path)
        matches = match_features(query_descriptors, dictionary_descriptors)

        avg_distance = np.mean([match.distance for match in matches])

        if avg_distance < min_distance:
            min_distance = avg_distance
            best_match_index = i
            best_matches = matches
            best_dictionary_keypoints = dictionary_keypoints

    if visualize:
        visualize_matches(query_img_path, dictionary_img_paths[best_match_index], query_keypoints, best_dictionary_keypoints, best_matches)

    return dictionary_img_paths[best_match_index]

# 辞書画像のパスとタグを用意します
dictionary_images = [
    {"path": "../dict_img/dedenne.jpg", "tag": "dedenne"},
    {"path": "../dict_img/pikachu.png", "tag": "pikachu"}
]

query_image_path = "../query_img/dedenne.jpg"

# 最も類似する辞書画像を見つける
best_match_path = find_best_match(query_image_path, [img["path"] for img in dictionary_images], visualize=True)

# タグを返す
for img in dictionary_images:
    if img["path"] == best_match_path:
        print("Best match tag:", img["tag"])
        break
