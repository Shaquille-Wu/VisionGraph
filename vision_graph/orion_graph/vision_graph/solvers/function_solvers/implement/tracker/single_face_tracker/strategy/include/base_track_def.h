//
// Created by yuan on 18-4-25.
//

#ifndef GRAPH_ROBOT_TRACKING_BASE_TRACK_DEF_H
#define GRAPH_ROBOT_TRACKING_BASE_TRACK_DEF_H

#include <opencv2/opencv.hpp>
#include "rpy_pose.h"

typedef enum
{
    TRACKING_SUCCESS = 1,
    TRACKING_UNCERTAIN = 0,
    TRACKING_FAILED = -1
} TrackingStatus;


typedef struct
{
    int track_id;
    cv::Rect pos;
    RPYPose attitude;
    TrackingStatus status;
} TargetState;


typedef enum
{
    ACTION_NONE = 0,
    ACTION_TRACK,
    ACTION_DET,
    ACTION_REID,
} TrackingAction;


typedef enum
{
    FRONT_FACE = 0,
    SIDE_FACE
} FaceDirection;


typedef enum
{
    FACE_TYPE_GOOD = -1,
    FACE_TYPE_NONE = -2,
    FACE_TYPE_SMALL = -3,
    FACE_TYPE_SIDE = -4,
    FACE_TYPE_OBSCURE = -5,
    FACE_TYPE_BLUR = -6,
    FACE_TYPE_INCOMPLETE = -7,
    FACE_TYPE_LOW_CONF = -8
} RejectFaceType;


#endif //ROBOT_TRACKING_BASE_TRACK_DEF_H
