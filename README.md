# An End-To-End Tracker

This is an end-to-end network which combines detector and matcher into one single network. Our target is to design an **End-to-End network** for detection and tracking.

## RoadMap

| Date   | Event                                |
| ------ | ------------------------------       |
| 201905 | Can Train on A-MOT dataset           |
| 201904 | Can Train On Whole UA-DETRAC dataset |
| 201904 | Recording Five Cities                |
| 201903 | Start A Plan of Create New Dataset   |
| 201902 | Optimized this network               |
| 201812 | Can Do the Basic Detection           |
| 201811 | Design the Loss Fucntion             |
| 201810 | Try the UA-DETRAC dataset            |
| 201809 | Re-design the input and output       |
| 201808 | Design the whole network             |
| 201807 | Start this idea                      |


## Issues
|   Symbol  | Meanings   |
| :-------: | :--------: |
| :hourglass_flowing_sand:      | Plan to solve         |
| :repeat:                      | try to solve it again |
| :no_entry:                    | abandoned issue       |
| :ballot_box_with_check:       | solved                |
| :black_square_button:         | unsolved              |
| :negative_squared_cross_mark: | cannot get solved     |
| :boom:                        | focusing              |
| :exclamation:                 | important             |

|   SartDate|                            Content                            | State |
| :------:  | :----------------------------------------------------------: | :---: |
| 2019/04/13  | Lost some objects<br/> ![](./images/progress/lost_objects1.png)![](./images/progress/lost_objects2.png)<br> |  :ballot_box_with_check: set confidence and existing threshold<br> :boom:20​19/04/13 process  |
| 2019/04/13  | NMS doesn't work well <br>![](./images/progress/nms_doesnt_work_well1.png) ![](./images/progress/nms_doesnt_work_well.png)<br> | 2019/04/13 :boom:  |
| 2019/04/13  | Problems of object at the edge of the frames. <br> ![](./images/progress/object_at_frame_edge.png)![](./images/progress/object_at_frame_edge1.png) |   2019/04/13 :boom:   |
| 2019/04/13  | Weird detected objects.<br> ![](./images/progress/werid_detect_object.png)![](./images/progress/werid_detect_object1.png) |   2019/04/13 :boom:   |


## Protocol
- bbox: the format is *(left, top, right, bottom)*
- $N_{tr}$: the track number.
- $N_{ba}$: the batch number.
- $N_{ti}$: the selected frame number
- $N_{pr}$: the number of prior boxes
- $W, H$: the input network image size (W, H).
- $W_{re}, H_{re}$: the real input image size.
- $F_t$: the $t^{th}$ frame
- $N_{fn}$: the input frame number
- $f(\cdot)$ is the operation to convert parameter to bboxes

## Network

The framework of our net is as following:

![framework](./images/framework.png)

## Loss Function

## Framework


## Requirement
|name           |version
|:---:          |:---:
|python         |3.6
|cuda           |8.0

Besides, install all the python package by following command

```shell
cd <project path>
pip install -r requiement.txt
```

## Preparation
- Download this code
- Download the [pre-trained base net model](https://drive.google.com/open?id=1CYb-RBZpz3UTbQRM4oIRipZrWrq10iIQ)

## Train

## Test

## Citation

## Copyright
