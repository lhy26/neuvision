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

#include "zcameraacquisition_fwd.h"
#include "zcameraacquisition_global.h"

#include "zwidget.h"


namespace Z3D
{

namespace Ui {
class ZCameraPreviewer;
}

class Z3D_CAMERAACQUISITION_SHARED_EXPORT ZCameraPreviewer : public ZWidget
{
    Q_OBJECT
    
public:
    explicit ZCameraPreviewer(QWidget *parent = nullptr);
    explicit ZCameraPreviewer(ZCameraPtr camera, QWidget *parent = nullptr);
    ~ZCameraPreviewer();

public slots:
    void setCamera(Z3D::ZCameraPtr camera);

protected slots:
    void closeEvent(QCloseEvent *);

    void onCameraRunningChanged();

private slots:
    void on_playButton_clicked();

    void on_snapshotButton_clicked();

private:
    Ui::ZCameraPreviewer *ui;

    Z3D::ZCameraPtr m_camera;
};

} // namespace Z3D
