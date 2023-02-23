import sys
sys.path.append('./LPRNet')
import cv2
import warnings
import logging
from utils.datasets import *
from utils.general import *
from utils.plots import plot_one_box_PIL
from utils.torch_utils import load_classifier

from LPRNet.model.LPRNET import LPRNet, CHARS
from LPRNet.model.STN import STNet
from LPRNet.LPRNet_Test import *

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Detect(object):
    def __init__(self,weights):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights=weights
        self.model = torch.load(self.weights, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()

        self.lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(self.device)
        self.lprnet.load_state_dict(
            torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
        self.lprnet.eval()

        self.STN = STNet()
        self.STN.to(self.device)
        self.STN.load_state_dict(
            torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
        self.STN.eval()
        logger.info('Successful to build Network!')
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect_image(self, img):
        since = time.time()
        im0 = img.copy()
        image = im0
        half = self.device.type != 'cpu'
        if half:
            self.model.half()
        img = letterbox(im0, new_shape=(384, 384))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.35, 0.5)
        return_boxs = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    return_boxs.append(xyxy)
                    bboxes = np.asarray(return_boxs)
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i, :4]
                        x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
                        w = int(x2 - x1 + 1.0)
                        h = int(y2 - y1 + 1.0)
                        img_box = np.zeros((h, w, 3))
                        img_box = image[y1:y2 + 1, x1:x2 + 1, :]
                        im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
                        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
                        data = torch.from_numpy(im).float().unsqueeze(0).to(self.device)  # torch.Size([1, 3, 24, 94])
                        transfer = self.STN(data)
                        preds = self.lprnet(transfer)
                        preds = preds.cpu().detach().numpy()  # (1, 68, 18)
                        labels, pred_labels = decode(preds, CHARS)
                        imgg=plot_one_box_PIL(xyxy,image,color=(255,0,0),label=labels[0],line_thickness=3)

        logger.info("model inference in {:2.3f} seconds".format(time.time() - since))
        cv2.imwrite('demo.jpg',imgg)
        cv2.imshow('image', imgg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO & LPR Demo')
    parser.add_argument("-image_path", default='test/4.jpg', type=str)
    parser.add_argument("-weights",default='weights/last.pt',type=str)
    args = parser.parse_args()
    det=Detect(args.weights)
    img=cv2.imread(args.image_path)
    det.detect_image(img)
