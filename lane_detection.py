import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    mask_lanes = cv2.bitwise_or(mask_white, mask_yellow)
    lane_edges = cv2.Canny(mask_lanes, 50, 150)

    obstacle_edges = cv2.Canny(gray, 100, 200)
    
    kernel = np.ones((15, 15), np.uint8)
    inflated_obs = cv2.dilate(obstacle_edges, kernel)

    unified_raw = cv2.bitwise_or(lane_edges, inflated_obs)

    src = np.float32([
        [int(0.44 * w), int(0.52 * h)],
        [int(0.56 * w), int(0.52 * h)],
        [int(0.98 * w), int(0.90 * h)],
        [int(0.02 * w), int(0.90 * h)]
    ])

    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    
    cost_map_bev = cv2.warpPerspective(unified_raw, M, (w, h))

    frame_copy = frame.copy()
    cv2.polylines(frame_copy, [src.astype(np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("1. Original", frame_copy)
    cv2.imshow("2. Edge-Based Obstacles", inflated_obs)
    cv2.imshow("3. Final 2D Cost Map", cost_map_bev)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
