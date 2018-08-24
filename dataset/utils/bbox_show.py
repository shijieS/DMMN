import cv2

def show_bboxes(frame, bboxes, color=None, titles=None, alpha=0.3, time=None):
    overlay = frame.copy()
    output = frame.copy()

    for bbox in bboxes:
        cv2.rectangle(overlay,
                      tuple(bbox[:2].astype(int)),
                      tuple(bbox[-2:].astype(int)),
                      color,
                      -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    if time is not None:
        cv2.putText(output, str(time), (25, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if titles is not None:
        for i, title in enumerate(titles):
            cv2.putText(output,
                        title,
                        tuple(bboxes[i, :][:2].astype(int)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        color,
                        2)
    return output
