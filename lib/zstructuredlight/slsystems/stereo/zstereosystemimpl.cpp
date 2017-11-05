//
// Z3D - A structured light 3D scanner
// Copyright (C) 2013-2016 Nicolas Ulrich <nikolaseu@gmail.com>
//
// This file is part of Z3D.
//
// Z3D is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Z3D is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Z3D.  If not, see <http://www.gnu.org/licenses/>.
//

#include "zstereosystemimpl.h"

#include "zdecodedpattern.h"
#include "zgeometryutils.h"
#include "zpinhole/zopencvstereocameracalibration.h"
#include "zpinhole/zpinholecameracalibration.h"
#include "zsimplepointcloud.h"

#include <QAtomicInt>
#include <QDateTime>
#include <QDebug>
#include <QFuture>
#include <QtConcurrentMap>
#include <QtConcurrentRun>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace Z3D
{

struct ParallelFringeProcessingImpl
{
    ParallelFringeProcessingImpl(ZStereoSystemImpl *system, Z3D::ZSimplePointCloudPtr cloudPtr, QAtomicInt &i2, const cv::Mat &image, float maxValidDistanceThreshold)
        : stereoSystem(system)
        , m_undistortedRays(stereoSystem->m_undistortedRays)
        , m_cloud(cloudPtr)
        , m_cloudPoints(m_cloud->points)
        , m_atomicInteger(i2)
        , intensityImg(image)
        , m_maxValidDistanceThreshold(maxValidDistanceThreshold)
    {
        camLcal = new ZPinholeCameraCalibration(stereoSystem->m_calibration->cameraMatrix[0],
                stereoSystem->m_calibration->distCoeffs[0],
                stereoSystem->m_calibration->imageSize[0]
        );
        camRcal = new ZPinholeCameraCalibration(stereoSystem->m_calibration->cameraMatrix[1],
                stereoSystem->m_calibration->distCoeffs[1],
                stereoSystem->m_calibration->imageSize[1]
        );
    }

    void operator()(const std::vector<const std::vector<cv::Vec2f>*> &fringePoints)
    {
        /// vectors of pairs, we will order them using the first component of the pair
        std::vector< std::vector< std::pair<float, cv::Vec2f> > > orderedPoints;
        orderedPoints.resize(fringePoints.size());

        /// k == camera index
        for (size_t k=0, size=fringePoints.size(); k<size; ++k) {
            const auto &fringePointsVector = *fringePoints[k];

            auto &orderedFringePoints = orderedPoints[k];
            orderedFringePoints.resize(fringePointsVector.size());

            const auto &undistortedRays = m_undistortedRays[k];

            /// insert points using the "y" coordinates of the images projected on the epipolar plane
            auto oit = orderedFringePoints.begin();
            for (auto lit = fringePointsVector.begin(), litEnd = fringePointsVector.end();
                 lit != litEnd;
                 ++lit, ++oit) {
                const auto &point = *lit;
                const int &x = int(point[0]);
                const int &y = int(point[1]);
                const auto &rectifiedPoint = undistortedRays[ stereoSystem->indexForPixel(x, y) ];
                auto &pair = *oit;
                pair.first = rectifiedPoint[1]; /// y coordinate
                pair.second = point;
            }

            /// sort values using first element of the pair
            std::sort(orderedFringePoints.begin(), orderedFringePoints.end(), [](const auto arg1, const auto arg2) -> bool {
                return arg1.first < arg2.first;
            });
        }

        cv::Vec3d intersection,
                realRayOrigin, realRayDirection,
                meanRayOrigin, meanRayDirection,
                firstRayOrigin, firstRayDirection,
                secondRayOrigin, secondRayDirection;

        auto &orderedLeftPoints = orderedPoints[0];
        auto &orderedRightPoints = orderedPoints[1];

        auto lNextIterator = orderedLeftPoints.cbegin();
        auto lIterator = lNextIterator++;
        auto lEnd = orderedLeftPoints.cend();
        auto rNextIterator = orderedRightPoints.cbegin();
        auto rIterator = rNextIterator++;
        auto rEnd = orderedRightPoints.cend();
        while (lNextIterator != lEnd && rNextIterator != rEnd) {
            const auto &lPair = *lIterator;
            const auto &rPair = *rIterator;
            const auto &lNextPair = *lNextIterator;
            const auto &rNextPair = *rNextIterator;

            const auto &lY = lPair.first;
            const auto &rY = rPair.first;
            const auto &lNextY = lNextPair.first;
            const auto &rNextY = rNextPair.first;

            /// we go in ascending order
            if (lNextY < rNextY) {
                /// increment left iterators
                lIterator = lNextIterator++;
            } else {
                /// increment right iterators
                rIterator = rNextIterator++;
            }

            static const bool useSubPixel = false;
            auto usePoint = false;
            auto alpha = 1.,
                    beta = 0.,
                    range = 0.;

            /// we go in ascending order
            if (lY < rY) {
                /// we have to intersect r with l and lNext
                if (std::max(rY - lY, lNextY - rY) < m_maxValidDistanceThreshold) {
                    usePoint = true;
                    range = lNextY - lY;
                    beta = (rY - lY) / range;
                    alpha = 1. - beta;

                    /// valid points, intersect and add point
                    const auto &realPoint = rPair.second;
                    const auto &firstMeanPoint = lPair.second;
                    const auto &secondMeanPoint = lNextPair.second;

                    /// "world" rays are already in common coordinate space (i.e. rotated, traslated, normalized)
                    if (useSubPixel) {
                        camRcal->getWorldRayForSubPixel(realPoint[0], realPoint[1], realRayOrigin, realRayDirection);

                        camLcal->getWorldRayForSubPixel(firstMeanPoint[0], firstMeanPoint[1], firstRayOrigin, firstRayDirection);
                        camLcal->getWorldRayForSubPixel(secondMeanPoint[0], secondMeanPoint[1], secondRayOrigin, secondRayDirection);
                    } else {
                        camRcal->getWorldRayForPixel(int(round(realPoint[0])), int(round(realPoint[1])), realRayOrigin, realRayDirection);

                        camLcal->getWorldRayForPixel(int(round(firstMeanPoint[0])), int(round(firstMeanPoint[1])), firstRayOrigin, firstRayDirection);
                        camLcal->getWorldRayForPixel(int(round(secondMeanPoint[0])), int(round(secondMeanPoint[1])), secondRayOrigin, secondRayDirection);
                    }
                }
            } else {
                /// we have to intersect l with r and rNext
                if (std::max(lY - rY, rNextY - lY) < m_maxValidDistanceThreshold) {
                    usePoint = true;
                    range = rNextY - rY;
                    beta = (lY - rY) / range;
                    alpha = 1. - beta;

                    /// valid points, intersect and add point
                    const auto &realPoint = lPair.second;
                    const auto &firstMeanPoint = rPair.second;
                    const auto &secondMeanPoint = rNextPair.second;

                    /// "world" rays are already in common coordinate space (i.e. rotated, traslated, normalized)
                    if (useSubPixel) {
                        camLcal->getWorldRayForSubPixel(realPoint[0], realPoint[1], realRayOrigin, realRayDirection);

                        camRcal->getWorldRayForSubPixel(firstMeanPoint[0], firstMeanPoint[1], firstRayOrigin, firstRayDirection);
                        camRcal->getWorldRayForSubPixel(secondMeanPoint[0], secondMeanPoint[1], secondRayOrigin, secondRayDirection);
                    } else {
                        camLcal->getWorldRayForPixel(int(round(realPoint[0])), int(round(realPoint[1])), realRayOrigin, realRayDirection);

                        camRcal->getWorldRayForPixel(int(round(firstMeanPoint[0])), int(round(firstMeanPoint[1])), firstRayOrigin, firstRayDirection);
                        camRcal->getWorldRayForPixel(int(round(secondMeanPoint[0])), int(round(secondMeanPoint[1])), secondRayOrigin, secondRayDirection);
                    }
                }
            }

            if (usePoint) {
                /// origin should be the same (in the case of pinhole camera calibration)
                /// but we calculate in case sometime we use another type of calibration...
                meanRayOrigin    = alpha * firstRayOrigin
                                 + beta * secondRayOrigin;
                meanRayDirection = alpha * firstRayDirection
                                 + beta * secondRayDirection;

                /// calculate intersection point between rays
                /// TODO this can be made faster if we use disparity!?
                intersection = GeometryUtils::intersectLineWithLine3D(
                            realRayOrigin, realRayDirection,
                            meanRayOrigin, meanRayDirection);

                /// use atomicint to atomically get current value and increment
                const auto currentIndex = size_t(m_atomicInteger.fetchAndAddAcquire(1));

                auto &currentPoint = m_cloudPoints[currentIndex];

                /// use the point
                currentPoint[0] = float(intersection[0]);
                currentPoint[1] = float(intersection[1]);
                currentPoint[2] = float(intersection[2]);

                /// point color. we use an intensity image from the left camera
                /// this point is not exactly the intersection point, but it's close enough
                const auto &intensityImgPoint = lPair.second;
                const int x = int(intensityImgPoint[0]);
                const int y = int(intensityImgPoint[1]);
                if (intensityImg.channels() == 1) {
                    /// black and white
                    const auto iB = intensityImg.at<unsigned char>(y, x);
                    uint32_t rgb = (static_cast<uint32_t>(255) << 24 |
                                    static_cast<uint32_t>(iB)  << 16 |
                                    static_cast<uint32_t>(iB)  <<  8 |
                                    static_cast<uint32_t>(iB));
                    currentPoint[3] = *reinterpret_cast<float*>(&rgb);
                } else {
                    /// RGB colors
                    cv::Vec3b intensity = intensityImg.at<cv::Vec3b>(y, x);
                    uint32_t rgb = (static_cast<uint32_t>(255)              << 24 |
                                    static_cast<uint32_t>(intensity.val[0]) << 16 |
                                    static_cast<uint32_t>(intensity.val[1]) <<  8 |
                                    static_cast<uint32_t>(intensity.val[2]));
                    currentPoint[3] = *reinterpret_cast<float*>(&rgb);
                }
            }
        }
    }

    ZStereoSystemImpl *stereoSystem;
    Z3D::ZPinholeCameraCalibrationWeakPtr camLcal;
    Z3D::ZPinholeCameraCalibrationWeakPtr camRcal;
    const std::vector< std::vector<cv::Vec3d> > &m_undistortedRays;
    Z3D::ZSimplePointCloudPtr m_cloud;
    Z3D::ZSimplePointCloud::PointVector &m_cloudPoints;
    QAtomicInt &m_atomicInteger;
    cv::Mat intensityImg;
    float m_maxValidDistanceThreshold;
};




ZStereoSystemImpl::ZStereoSystemImpl(ZMultiCameraCalibrationPtr stereoCalibration, QObject *parent)
    : QObject(parent)
    , m_calibration(std::dynamic_pointer_cast<ZOpenCVStereoCameraCalibration>(stereoCalibration))
    , m_ready(false)
{
    m_undistortedRays.resize(2);
    //m_undistortedWorldRays.resize(2);
    m_R.resize(2);
    m_P.resize(2);

    m_imageSize = m_calibration->imageSize[0];

    QtConcurrent::run(this, &ZStereoSystemImpl::stereoRectify, 0);
}

ZStereoSystemImpl::~ZStereoSystemImpl()
{

}

bool ZStereoSystemImpl::ready() const
{
    return m_ready;
}

void ZStereoSystemImpl::stereoRectify(double alpha)
{
    qDebug() << Q_FUNC_INFO;

    setReady(false);

    /// we need calibrated cameras!
    if (!m_calibration) {
        qWarning() << "invalid calibration! this only works for stereo calibration!";
        return;
    }

    cv::Rect validRoi[2];

    cv::stereoRectify(m_calibration->cameraMatrix[0], m_calibration->distCoeffs[0],
                      m_calibration->cameraMatrix[1], m_calibration->distCoeffs[1],
                      m_imageSize,
                      m_calibration->R, m_calibration->T,
                      m_R[0], m_R[1],
                      m_P[0], m_P[1],
                      m_Q,
                      /** flags – Operation flags that may be zero or CV_CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area. */
                      0 /*CALIB_ZERO_DISPARITY*/,
                      /** alpha – Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Obviously, any intermediate value yields an intermediate result between those two extreme cases.*/
                      alpha,
                      m_imageSize, &validRoi[0], &validRoi[1]);

    /*
    QString timeStr = QDateTime::currentDateTime().toString("yyyy.MM.dd_hh.mm.ss");
    cv::FileStorage fs(qPrintable(QString("%1_stereoRectify.yml").arg(timeStr)), CV_STORAGE_WRITE);
    if( fs.isOpened() ) {
        fs << "R" << R << "T" << T << "R1" << m_R[0] << "R2" << m_R[1] << "P1" << m_P[0] << "P2" << m_P[1] << "Q" << m_Q;
        fs.release();
    } else {
        qCritical() << "Error: can not save the stereo rectification parameters";
    }*/

    QtConcurrent::run(this, &ZStereoSystemImpl::precomputeOptimizations);
}

Z3D::ZSimplePointCloudPtr ZStereoSystemImpl::triangulateOptimized(const cv::Mat &intensityImg,
                                                 const std::map<int, std::vector<cv::Vec2f> > &leftFringePoints,
                                                 const std::map<int, std::vector<cv::Vec2f> > &rightFringePoints,
                                                 int maxPosibleCloudPoints,
                                                 float maxValidDistanceThreshold)
{
    QTime startTime;
    startTime.start();

    Z3D::ZSimplePointCloudPtr cloud(new Z3D::ZSimplePointCloud());

    qDebug() << "maximum posible points in the cloud:" << maxPosibleCloudPoints;

    /// reserve maximum posible size, avoid reallocations
    cloud->width  = maxPosibleCloudPoints;
    cloud->height = 1;
    cloud->points.resize(size_t(cloud->width * cloud->height));

    QAtomicInt i = 0;

    qDebug() << "using maxValidDistanceThreshold" << maxValidDistanceThreshold;

    QVector<std::vector<const std::vector<cv::Vec2f>*> > parallellData;
    parallellData.resize(int(leftFringePoints.size()));

    const auto rightPointsItEnd = rightFringePoints.cend();
    int currentIndex = -1;
    for (auto it = leftFringePoints.cbegin(), itEnd = leftFringePoints.cend();
         it != itEnd;
         ++it) {

        const int fringeID = it->first;

        /// skip if fringe is only present in one camera
        const auto rightPointsIt = rightFringePoints.find(fringeID);
        if (rightPointsIt == rightPointsItEnd) {
            continue;
        }

        const auto &rightPointsVector = rightPointsIt->second;
        if (rightPointsVector.empty()) {
            continue;
        }

        const auto &leftPointsVector = it->second;

        currentIndex++;
        auto &fringePoints = parallellData[currentIndex];
        fringePoints.push_back( &leftPointsVector );
        fringePoints.push_back( &rightPointsVector );
    }

    parallellData.resize(currentIndex+1);

    qDebug() << "finished preparing data to process in" << startTime.elapsed() << "msecs";

    /// execution on parallel in other threads (from the thread pool)
    QFuture<void> future = QtConcurrent::map(parallellData, ParallelFringeProcessingImpl(this, cloud, i, intensityImg, maxValidDistanceThreshold));

    /// wait for all threads to finish
    future.waitForFinished();

    if (i>0) {
//        qDebug() << "sum ray error:" << sqrt(squaredDistanceSum) << "rms ray error:" << sqrt(squaredDistanceSum)/i;

        cloud->width  = i;
        cloud->height = 1;
        cloud->points.resize(size_t(cloud->width * cloud->height));

        return cloud;
    }

    /// user of this function must check if it is valid
    qWarning() << "The point cloud could not be calculated, it's empty! Returning invalid cloud.";
    return nullptr;
}

template<typename T>
ZSimplePointCloudPtr process(cv::Mat lProjMatrix, cv::Mat rProjMatrix, cv::Mat Q, cv::Mat leftImg, cv::Mat rightImg) {
    const cv::Size &imgSize = leftImg.size();
    const int &imgHeight = imgSize.height;
    const int &imgWidth = imgSize.width;

    //std::vector<cv::Vec2f> lPoints;
    //std::vector<cv::Vec2f> rPoints;
    std::vector<cv::Vec3f> disparity;

    for (int y=0; y<imgHeight; ++y) {
        const T* imgData = leftImg.ptr<T>(y);
        const T* rImgData = rightImg.ptr<T>(y);
        for (int x=0, rx=0; x<imgWidth; ++x, ++imgData) {
            if (*imgData == ZDecodedPattern::NO_VALUE) {
                continue;
            }

            for (; rx<imgWidth-1; ++rx, ++rImgData) {
                if (*rImgData == ZDecodedPattern::NO_VALUE) {
                    continue;
                }

                if (*imgData >= *rImgData && *imgData <= *(rImgData+1)) {
                    //lPoints.push_back(cv::Vec2f(x, y));
                    //rPoints.push_back(cv::Vec2f(rx, y));
                    disparity.push_back(cv::Vec3f(x, y, rx - x));
                    break;
                }
            }
        }
    }

    qDebug() << "found" << disparity.size() << "matches";

    std::vector<cv::Vec3f> points3f;
    cv::perspectiveTransform(disparity, points3f, Q);

    ZSimplePointCloudPtr pointCloud(new ZSimplePointCloud);
    auto &points = pointCloud->points;
    points.resize(points3f.size());
    for (size_t i=0; i<points.size(); ++i) {
        points[i][0] = points3f[i][0];
        points[i][1] = points3f[i][1];
        points[i][2] = points3f[i][2];
    }

    /*cv::Mat points3d;
    cv::triangulatePoints(lProjMatrix, rProjMatrix, lPoints, rPoints, points3d);

    ZSimplePointCloudPtr pointCloud(new ZSimplePointCloud);
    auto &points = pointCloud->points;
    points.resize(points3d.cols);
    for (int i=0; i<points.size(); ++i) {
        points[i][0] = points3d.at<float>(0, i);
        points[i][1] = points3d.at<float>(1, i);
        points[i][2] = points3d.at<float>(2, i);
        points[i] /= points3d.at<float>(3, i);
    }*/

    return pointCloud;
}

ZSimplePointCloudPtr ZStereoSystemImpl::triangulate(const cv::Mat &colorImg, const cv::Mat &leftDecodedImage, const cv::Mat &rightDecodedImage)
{
    QTime time;
    time.start();

    cv::Mat rmap[2][2];
    for (int k = 0; k < 2; k++) {
        cv::initUndistortRectifyMap(m_calibration->cameraMatrix[k], m_calibration->distCoeffs[k], m_R[k], m_P[k], m_imageSize, CV_16SC2, rmap[k][0], rmap[k][1]);
    }

    cv::Mat leftRemapedImage;
    cv::remap(leftDecodedImage, leftRemapedImage, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
    cv::Mat rightRemapedImage;
    cv::remap(rightDecodedImage, rightRemapedImage, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

    qDebug() << "rectification finished in" << time.elapsed() << "msecs";

    switch (leftRemapedImage.type()) {
    case CV_32F: // float
        return process<float>(m_P[0], m_P[1], m_Q, leftRemapedImage, rightRemapedImage);
        break;
    default:
        qWarning() << "unkwnown image type:" << leftRemapedImage.type();
    }

    return nullptr;
}

void ZStereoSystemImpl::setCalibration(ZStereoCameraCalibrationPtr calibration)
{
    m_calibration = calibration;
}

void ZStereoSystemImpl::precomputeOptimizations()
{
    qDebug() << "precomputing optimizations...";

    QTime time;
    time.start();

    const int &width = m_imageSize.width;
    const int &height = m_imageSize.height;

    const auto pixelCount = size_t(width * height);

    /// fill "distorted" points for every pixel in the image
    std::vector<cv::Point2d> points;
    points.resize( pixelCount );
    for (int iy = 0; iy < height; ++iy) {
        for (int ix = 0; ix < width; ++ix) {
            points[ indexForPixel(ix, iy) ] = cv::Point2d(ix, iy);
        }
    }

    /// undistorted points will be returned here
    std::vector<cv::Point2d> undistortedPoints;
    undistortedPoints.resize( pixelCount );

    for (size_t k=0; k<2; k++) {
        qDebug() << "initializing undistorted rays lookup table for camera" << k << "with" << pixelCount << "values";

        /// undistortedPoints will be in world space
        cv::undistortPoints(points, undistortedPoints, m_calibration->cameraMatrix[k], m_calibration->distCoeffs[k], m_R[k]/*, m_P[k]*/);
        //cv::undistortPoints(points, undistortedPoints, mCal[k]->cvCameraMatrix(), mCal[k]->cvDistortionCoeffs(), m_R[k].t());

        auto &m_undistortedRays_k = m_undistortedRays[k];
        //std::vector<cv::Vec3d> &m_undistortedWorldRays_k = m_undistortedWorldRays[k];

        m_undistortedRays_k.resize( pixelCount );
        //m_undistortedWorldRays_k.resize( pixelCount );

        //cv::Matx33d R(m_R[k]);
        //cv::Vec3d ray;

        for (int iy = 0; iy < height; ++iy) {
            for (int ix = 0; ix < width; ++ix) {
                const auto index = indexForPixel(ix, iy);

                const auto &undistortedPoint = undistortedPoints[index];

                auto &ray = m_undistortedRays_k[index];
                ray[0] = undistortedPoint.x;
                ray[1] = undistortedPoint.y;
                ray[2] = 1.;

                // store undistorted ray
                // don't normalize because we use the "y" coordinate over the epipolar plane
                // to optimize matches between left image and right image
                //m_undistortedRays_k[index] = ray; // cv::normalize(ray);

                //m_undistortedWorldRays_k[index] = /*mCal[k]->rotation() * R * */ ray;
                //m_undistortedWorldRays[k][index] = mCal[k]->rotation() * R.t() * ray;
            }
        }

        qDebug() << "finished initialization of undistorted rays lookup table for camera" << k;
    }

    qDebug() << "finished precomputing optimizations in" << time.elapsed() << "msecs";

    setReady(true);
}

void ZStereoSystemImpl::setReady(bool arg)
{
    if (m_ready == arg) {
        return;
    }

    m_ready = arg;
    emit readyChanged(arg);
}

} // namespace Z3D
