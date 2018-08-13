import torch as t
from torch import nn
from torch.nn import functional as F
from utils import array_tool as at


class FasterRCNN(nn.Module):
    """Base class for Faster RCNN

    This is a base class for Faster R-CNN links supporting object detection

    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that belong to the proposal RoIs, classify the categories of the objects in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`torch.nn.Module` objects :obj:`feature`, :obj:`rpn` and :obj:`head`

    There are two function :meth:`predict` and :meth:`__call__` to conduct object detection
    :meth:`predict` takes images and returns bounding boxes that are converted to image coordinates. This will be useful for a scenario when Faster R-CNN is treated as a black box function, for instance.
    :meth:`__call__` is provided for a scenario when intermediate outputs are needed, for instance, for training and debugging.

    Link that support object detection API have method :meth:`predict` with the same interface. Please refer to :meth:`predict` for further details.

    ..[#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian sun. Faster R-CNN: Towards Real-time object detection with region proposal network. NIPS 2015.

    Args:
        extractor (nn.Module): A module that takes a BCHW iamge array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as :class:`model.region_proposal_network.RegionProposalNetwork`. Please refer to the documentation found there.
        head (nn.Module): A module that takes a BCHW variable, RoIs and batch indices for RoIs. This returns class dependent localization parameters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of localization estimates.
        local_normalize_std (tuple of four floats): Standard deviation of localization estimates.


    """

    def __init__(self, extractor, rpn, head,
                 local_normalize_mean = (0., 0., 0., 0.),
                 local_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()

        self.extractor = extractor
        self.rpn = rpn
        self.head = head


        self.local_normalize_mean = local_normalize_mean
        self.local_normalize_std = local_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        """ Forward Faster R-CNN

        Scaling parameter :obj:`scale` is used by RPN to determine the threshold to select small objects, which are going to be rejected irrespective of their confidence scores.

        Here are notations used.

        * :math: `N` is the number of batch size
        * :math: `R` is the total number of RoIs produced across batches. Given :math:`R_i` proposed RoIs from the :math:`i` th image, :math:`R =\\sum_{i=1}^N R_i`
        * :math:`L` is the number of classes excluding the background.

        Classes are ordered by the background, the first class, ..., and the :math:`L` th class.

        Args:
            x (autograd.Variable): 4D image variable.
            scale (float): Amount of scaling applied to the raw image during preprocessing.

        Returns:
            Variable, Variable, array, array:
            Return tuple of four values list below.

            * **roi_cls_locs**: Offsets and scalings for the proposed RoIs. Its shape is :math:`(R, (L+1)\\times 4)`.
            * **roi_scores**: Class predictions for the proposed RoIs. Its shape is :math:`(R, L+1)`.
            * **rois**: RoIs proposed by RPN. Its shape is :math:`(R,4)`
            * **roi_indices**: Batch indices of RoIs. Its shape is :math:`(R, )`.
        """
        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)

        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and :obj:`self.score_thresh`. These values are a threshold value used for non maximum suppression and a threshold value to discard low confidence proposal in :meth:`predict`, respectively.

        If the attribute need to be changed to something other than the values provided in the presets, please modify them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the preset to use
        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs, size=None, visualize=False):
        """Detect objects from images.

        This method predicts objects from each image.

        Args:
            imgs (iterable of numpy.ndarray): Array holding images. All images are in CHW and RGB format and the range of their value is :math:`[0,255]`.

        Returns:
            tuple of lists:
            This method returns a tuple of three lists,
            :obj:`(bboxes, labels, scores)`.

            * **bboxes** : A list of float arrays of shape :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in a images. Each bounding boxes is organized by \
                :math:`(y_{min}, x_{min}, y_{max}, x_{max}`\
                in the second axis.
            * **labels** : A list of integer arrays of shape :math:`(R,)`. Each value indicates the class of the bounding box. Values in the range :math:`[0, L-1]`, where :math:`L` is the number of the foreground classes.
            * **scores** : A list of float arrays of shape :math:`(R,)`. Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = t.autograd.Variable(at.totensor(img).float()[None], volatile=True)
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)

            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # convert predictions to bounding boxes in image coordinates
            # bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.local_normalize_mean).cuda.repeat(self.n_class)[None]
            std = t.Tensor(self.local_normalize_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape(-1, 4))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            prob = at.tonumpy(F.softmax(at.tovariable(roi_scores), dim=1))
            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores
