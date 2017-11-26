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

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "zsimplepointcloud.h"
#include "zpatternprojection.h"
#include "zpatternprojectionprovider.h"
#include "zstructuredlightsystem.h"
#include "zstructuredlightsystemprovider.h"

#include "src/zpointcloudwidget.h"

#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QTimer>



MainWindow::MainWindow(QWidget *parent)
    : Z3D::ZMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_currentStructuredLightSystem(nullptr)
{
    ui->setupUi(this);

    m_pointCloudWidget = new ZPointCloudWidget(this);
    ui->glLayout->addWidget(m_pointCloudWidget);

    /// disable all until ready
    ui->centralwidget->setEnabled(false);

    /// finish initialization
    QTimer::singleShot(100, this, &MainWindow::init);
}

MainWindow::~MainWindow()
{
    qDebug() << "deleting ui...";
    delete ui;

    qDebug() << Q_FUNC_INFO << "finished";
}

void MainWindow::init()
{
    QDir configDir = QDir::current();
#if defined(Q_OS_WIN)
//        if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
//            pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
    if (configDir.dirName() == "MacOS") {
        configDir.cdUp();
        configDir.cdUp();
        configDir.cdUp();
    }
#endif

    QString settingsFile = configDir.absoluteFilePath(QString("%1.ini").arg(QApplication::applicationName()));
    qDebug() << "trying to load config from:" << settingsFile;
    QSettings settings(settingsFile, QSettings::IniFormat);

    m_currentStructuredLightSystem = Z3D::ZStructuredLightSystemProvider::get(&settings);

    if (m_currentStructuredLightSystem) {
        connect(m_currentStructuredLightSystem.get(), &Z3D::ZStructuredLightSystem::readyChanged,
                ui->centralwidget, &QWidget::setEnabled);
        connect(m_currentStructuredLightSystem.get(), &Z3D::ZStructuredLightSystem::scanFinished,
                this, &MainWindow::onScanFinished);
        connect(ui->startAcquisitionButton, &QPushButton::clicked,
                m_currentStructuredLightSystem.get(), &Z3D::ZStructuredLightSystem::start);

        /// add config widget
        QWidget *currentWidget = Z3D::ZStructuredLightSystemProvider::getConfigWidget(m_currentStructuredLightSystem.get());
        ui->structuredLightSystemConfigLayout->addWidget(currentWidget);
        currentWidget->setVisible(true);

        ui->structuredLightSystemLabel->setText(m_currentStructuredLightSystem->metaObject()->className());

        /// add pattern projection config widget
        auto m_currentPatternProjection = m_currentStructuredLightSystem->patternProjection();
        ui->patternTypeLabel->setText(m_currentPatternProjection->metaObject()->className());

        QWidget *currentWidget2 = Z3D::ZPatternProjectionProvider::getConfigWidget(m_currentPatternProjection.get());
        ui->patternProjectionConfigLayout->addWidget(currentWidget2);
        currentWidget2->setVisible(true);
    }
}

void MainWindow::closeEvent(QCloseEvent * /*event*/)
{
    /// closing the main window causes the program to quit
    qApp->quit();
}

void MainWindow::onScanFinished(Z3D::ZSimplePointCloudPtr cloud)
{
    if (!cloud) {
        QMessageBox::warning(this, tr("Scan error"),
                             tr("The point cloud is empty.\nIs everything configured correctly?\nIs there anything inside the scan volume?"));
        return;
    }

    m_pointCloudWidget->setPointCloudData(ZPointCloudData::Ptr(new ZPointCloudData(cloud)));

    /*
    QString filename = QString("%1/pointCloud.pcd").arg(decodedPattern->scanTmpFolder);
    qDebug() << "saving point cloud data to" << filename;
    pcl::io::savePCDFile(qPrintable(filename), *cloud);
    */
    /// FIXME
    /// getCloudViewer()->addPointCloud(cloud);


    QDir folder = QDir::current();
#if defined(Q_OS_MAC)
    if (folder.dirName() == "MacOS") {
        folder.cdUp();
        folder.cdUp();
        folder.cdUp();
    }
#endif

    QString fileName = QFileDialog::getSaveFileName(
                this,
                tr("Save point cloud"),
                QString("%1/%2_pointCloud.asc")
                    .arg(folder.absolutePath())
                    .arg(QDateTime::currentDateTime().toString("yyyyMMddhhmmss")),
                tr("ASCII file (*.asc);;Point Cloud Library file (*.pcd)"));

    if (fileName.isEmpty() || fileName.isNull()) {
        return;
    }

    QFile file(fileName);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream fileTextStream(&file);
        if (fileName.endsWith(".pcd", Qt::CaseInsensitive)) {
            /// if trying to save to PCD format
            fileTextStream << QString(
                              "# .PCD v.7 - Point Cloud Data file format\n"
                              "VERSION .7\n"
                              "FIELDS x y z rgb\n"
                              "SIZE 4 4 4 4\n"
                              "TYPE F F F F\n"
                              "COUNT 1 1 1 1\n"
                              "WIDTH %1\n"
                              "HEIGHT 1\n"
                              "VIEWPOINT 0 0 0 1 0 0 0\n"
                              "POINTS %1\n"
                              "DATA ascii\n")
                              .arg(cloud->points.size());
            for (const auto &point : cloud->points) {
                fileTextStream << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << "\n";
            }
        } else if (fileName.endsWith(".asc", Qt::CaseInsensitive)) {
            /// ASCII file format
            for (const auto &point : cloud->points) {
                fileTextStream << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << "\n";
            }
        } else {
            QMessageBox::warning(this, tr("Error saving point cloud"),
                                 tr("Invalid file %1\nNothing saved.").arg(fileName));
        }
        fileTextStream.flush();
        file.flush();
        file.close();
        qDebug() << "point cloud written to" << file.fileName();
    } else {
        qCritical() << "cant open file to write fringePoints" << file.fileName();
    }
}
