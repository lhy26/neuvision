/* * Z3D - A structured light 3D scanner
 * Copyright (C) 2013-2016 Nicolas Ulrich <nikolaseu@gmail.com>
 *
 * This file is part of Z3D.
 *
 * Z3D is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Z3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Z3D.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "zstructuredlight_fwd.h"

#include <Z3DCameraCalibration>

#include <QObject>

#include <opencv2/core/mat.hpp>

namespace Z3D
{

struct ParallelFringeProcessingImpl;

class ZStereoSystemImpl : public QObject
{
    Q_OBJECT

    Q_PROPERTY(bool ready READ ready WRITE setReady NOTIFY readyChanged)

public:
    friend struct ParallelFringeProcessingImpl;

    explicit ZStereoSystemImpl(ZMultiCameraCalibrationPtr stereoCalibration, QObject *parent = nullptr);
    ~ZStereoSystemImpl();

    bool ready() const;

signals:
    void readyChanged(bool arg);

public slots:
    void stereoRectify(double alpha = -1);

//    cv::Mat getRectifiedSnapshot2D();
    //Z3D::ZSimplePointCloud::Ptr getRectifiedSnapshot3D(int cameraIndex, float imageScale = 50, int lod_step = 4);

    Z3D::ZSimplePointCloudPtr triangulateOptimized(const cv::Mat &intensityImg,
            const std::map<int, std::vector<cv::Vec2f> > &leftPoints,
            const std::map<int, std::vector<cv::Vec2f> > &rightPoints,
            int maxPosibleCloudPoints,
            float maxValidDistanceThreshold);

    Z3D::ZSimplePointCloudPtr triangulate(const cv::Mat &colorImg,
                                          const cv::Mat &leftDecodedImage,
                                          const cv::Mat &rightDecodedImage);

    void setCalibration(ZStereoCameraCalibrationPtr calibration);

protected slots:
    void precomputeOptimizations();

    inline size_t indexForPixel(int x, int y) const { return size_t(x + y * m_imageSize.width); }

    void setReady(bool arg);

protected:
    ZStereoCameraCalibrationPtr m_calibration;

    std::vector< std::vector<cv::Vec3d> > m_undistortedRays;
    //std::vector< std::vector<cv::Vec3d> > m_undistortedWorldRays;

    cv::Size m_imageSize;

    std::vector<cv::Mat> m_R;
    std::vector<cv::Mat> m_P;
    cv::Mat m_Q;

private:
    bool m_ready;
};

} // namespace Z3D
