import cv2

def plot_result(result):
    img = result.plot()
    cv2.imshow('Detection Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()