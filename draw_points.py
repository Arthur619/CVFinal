import cv2


def kp_drawer(result):
    keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = keypoints.cpu().numpy()  # convert to numpy array
    # draw keypoints, set first keypoint is red, second is blue
    for keypoint in keypoints.data:
        points=[]
        order=[[0, 1],[0,2] ,[1, 2],[0,14],[14,5],[14,3],[14,13],[5,6],[3,4],[13,10],[13,7],[7,8],[7,9],[10,11],[10,12]]
        for i in range(len(keypoint)):
            x, y ,_ = keypoint[i]
            x, y = int(x), int(y)
            points.append((x, y))  # 将点的坐标添加到列表中
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        points = [(points[i[0]], points[i[1]]) for i in order]
        for point in points:
            cv2.line(image, (int(point[0][0]), int(point[0][1])), (int(point[1][0]), int(point[1][1])), (0, 255, 0), 2)
    return image
#%%
result = list(model(image_path,  stream=True))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
boxes = result.boxes  # Boxes object for bbox outputs
boxes = boxes.cpu().numpy()  # convert to numpy array
objs_labels = model.names  # get class labels
image = cv2.imread(image_path)
# 遍历每个框
for box in boxes.data:
    l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom
    conf, id = box[4:]  # confidence, class
    id = int(id)
    # 绘制框
    cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
    # 绘制类别+置信度（格式：98.1%）
    cv2.putText(image, f"{objs_labels[id]} {conf * 100:.1f}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1)
# 遍历keypoints
image=kp_drawer(result)