#include <QApplication>
#include <QQmlApplicationEngine>
#include <QSurfaceFormat>

#include "qpointcloud.h"
#include "qpointcloudgeometry.h"
#include "qpointfield.h"
#include "qpointcloudreader.h"

int main(int argc, char *argv[])
{
    QSurfaceFormat glFormat;
    glFormat.setVersion(3, 3);
    glFormat.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(glFormat);

    QGuiApplication app(argc, argv);

    qmlRegisterType<QPointCloudReader>("pcl", 1, 0, "PointcloudReader");
    qmlRegisterType<QPointcloud>("pcl", 1, 0, "Pointcloud");
    qmlRegisterType<QPointcloudGeometry>("pcl", 1, 0, "PointcloudGeometry");
    qmlRegisterUncreatableType<QPointfield>("pcl", 1, 0, "Pointfield", "Can not yet be created in qml, use PointcloudReader.");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));

    return app.exec();
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hinst, HINSTANCE, LPSTR argv, int argc)
{
//    int argc=1;
//    char *argv[] = {"temp"};
    return main(argc, &argv);
}
#endif
